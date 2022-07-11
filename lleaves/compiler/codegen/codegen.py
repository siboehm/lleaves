from dataclasses import dataclass

from llvmlite import ir

from lleaves.compiler.utils import ISSUE_ERROR_MSG, MissingType

BOOL = ir.IntType(bits=1)
DOUBLE = ir.DoubleType()
FLOAT = ir.FloatType()
INT_CAT = ir.IntType(bits=32)
INT = ir.IntType(bits=32)
LONG = ir.IntType(bits=64)
ZERO_V = ir.Constant(BOOL, 0)
FLOAT_POINTER = ir.PointerType(FLOAT)
DOUBLE_PTR = ir.PointerType(DOUBLE)


def iconst(value):
    assert -(2**31) <= value <= 2**31 - 1
    return ir.Constant(INT, value)


def lconst(value):
    assert -(2**63) <= value <= 2**63 - 1
    return ir.Constant(LONG, value)


def fconst(value):
    return ir.Constant(FLOAT, value)


def dconst(value):
    return ir.Constant(DOUBLE, value)


@dataclass
class LTree:
    """Class for the LLVM function of a tree paired with relevant non-LLVM context"""

    llvm_function: ir.Function
    class_id: int


def gen_forest(forest, module, fblocksize, froot_func_name):
    """
    Populate the passed IR module with code for the forest.

    Overview of generated IR
    ---------------------------
    The forest is represented by the @forest_root function, which is called from Python.
    This function loops over every row of the input array. For each row:
    - Load all attributes, cast categorical attributes to INT.
    - Iteratively call each @tree_<index> function. This function returns a DOUBLE. The results
      of all @tree_<index> calls are summed up into a result variable.
    - The final result variable is run through the objective function (eg sigmoid)
      and stored in the results array passed by the caller.

    The actual IR is slightly more complicated, because lleaves implements instruction cache blocking.
    The full set of tree_functions is divided into chunks, with each chunk containing a subset of tree_functions.
    For each chunk we process every row of the input array in sequence which minimizes icache misses.
    Pseudo-code for tree with 100 tree_functions and chunks of size 50:
    for tree in tree_funcs[0:50]:
        for row in range(len(input)):
           result[row] += tree(input[row])
    for tree in tree_funcs[50:100]:
        for row in range(len(input)):
           result[row] += tree(input[row])
    ...

    For each tree in the forest there is a @tree_<index> function which takes all attributes as arguments

    For each node there are 0-2 blocks in the @tree_<index> function.
    - Decision node (categorical): 2 Blocks, 1 for the node, 1 for the categorical bitset-comparison
    - Decision node (numerical): 1 Block.
    - Leaf node: 0-1 Blocks. If a decision node has only leaves as children we fuse both leaves into
      a single switch instr in the decision node's block.
    Each node cbranches to the child node's block.

    :return: None
    """

    # entry function called from Python
    root_func = ir.Function(
        module,
        ir.FunctionType(ir.VoidType(), (DOUBLE_PTR, DOUBLE_PTR, INT, INT)),
        name=froot_func_name,
    )

    def make_tree(tree):
        # declare the function for this tree
        func_dtypes = (INT_CAT if f.is_categorical else DOUBLE for f in tree.features)
        scalar_func_t = ir.FunctionType(DOUBLE, func_dtypes)
        tree_func = ir.Function(module, scalar_func_t, name=str(tree))
        tree_func.linkage = "private"
        # populate function with IR
        gen_tree(tree, tree_func)
        return LTree(llvm_function=tree_func, class_id=tree.class_id)

    tree_funcs = [make_tree(tree) for tree in forest.trees]

    if forest.n_classes > 1:
        # better locality by running trees for each class together
        tree_funcs.sort(key=lambda t: t.class_id)

    _populate_forest_func(forest, root_func, tree_funcs, fblocksize)


def gen_tree(tree, tree_func):
    """generate code for tree given the function, recursing into nodes"""
    node_block = tree_func.append_basic_block(name=str(tree.root_node))
    gen_node(tree_func, node_block, tree.root_node)


def gen_node(func, node_block, node):
    """generate code for node, recursing into children"""
    if node.is_leaf:
        _gen_leaf_node(node_block, node)
    else:
        _gen_decision_node(func, node_block, node)


def _gen_leaf_node(node_block, leaf):
    """populate block with leaf's return value"""
    builder = ir.IRBuilder(node_block)
    builder.ret(dconst(leaf.value))


def _gen_decision_node(func, node_block, node):
    """generate code for decision node, recursing into children"""
    builder = ir.IRBuilder(node_block)

    # optimization for node where both children are leaves (switch instead of cbranch)
    is_fused_double_leaf_node = node.left.is_leaf and node.right.is_leaf
    if is_fused_double_leaf_node:
        left_block = None
        right_block = None
        # categorical nodes have a fastpath which can branch-right early
        # so they still need a right block
        if node.decision_type.is_categorical:
            right_block = func.append_basic_block(name=str(node.right))
    else:
        left_block = func.append_basic_block(name=str(node.left))
        right_block = func.append_basic_block(name=str(node.right))

    # populate this node's block up to the terminal statement
    if node.decision_type.is_categorical:
        bitset_comp_block = builder.append_basic_block(str(node) + "_cat_bitset_comp")
        bitset_builder = ir.IRBuilder(bitset_comp_block)
        comp = _populate_categorical_node_block(
            func, builder, bitset_builder, node, bitset_comp_block, right_block
        )
        builder = bitset_builder
    else:
        comp = _populate_numerical_node_block(func, builder, node)

    # finalize this node's block with a terminal statement
    if is_fused_double_leaf_node:
        ret = builder.select(comp, dconst(node.left.value), dconst(node.right.value))
        builder.ret(ret)
    else:
        builder.cbranch(comp, left_block, right_block)

    # populate generated child blocks
    if left_block:
        gen_node(func, left_block, node.left)
    if right_block:
        gen_node(func, right_block, node.right)


def _populate_instruction_block(
    forest, root_func, tree_funcs, setup_block, next_block, eval_obj_func
):
    """Generates an instruction_block: loops over all input data and evaluates its chunk of tree_funcs."""
    data_arr, out_arr, start_index, end_index = root_func.args

    # -- SETUP BLOCK
    builder = ir.IRBuilder(setup_block)
    start_index = builder.zext(start_index, LONG)
    end_index = builder.zext(end_index, LONG)
    loop_iter = builder.alloca(LONG, 1, "loop-idx")
    builder.store(start_index, loop_iter)
    condition_block = root_func.append_basic_block("loop-condition")
    builder.branch(condition_block)
    # -- END SETUP BLOCK

    # -- CONDITION BLOCK
    builder = ir.IRBuilder(condition_block)
    comp = builder.icmp_signed("<", builder.load(loop_iter), end_index)
    core_block = root_func.append_basic_block("loop-core")
    builder.cbranch(comp, core_block, next_block)
    # -- END CONDITION BLOCK

    # -- CORE LOOP BLOCK
    builder = ir.IRBuilder(core_block)
    # build args arr, convert categoricals vars from float to int
    args = []
    loop_iter_reg = builder.load(loop_iter)

    n_args = ir.Constant(LONG, forest.n_args)
    iter_mul_nargs = builder.mul(loop_iter_reg, n_args)
    idx = (builder.add(iter_mul_nargs, lconst(i)) for i in range(forest.n_args))
    raw_ptrs = [builder.gep(root_func.args[0], (c,)) for c in idx]
    # cast the categorical inputs to integer
    for feature, ptr in zip(forest.features, raw_ptrs):
        el = builder.load(ptr)
        if feature.is_categorical:
            # first, check if the value is NaN
            is_nan = builder.fcmp_ordered("uno", el, dconst(0.0))
            # if it is, return smallest possible int (will always go right), else cast to int
            el = builder.select(is_nan, iconst(-(2**31)), builder.fptosi(el, INT_CAT))
            args.append(el)
        else:
            args.append(el)
    # iterate over each tree, sum up results
    results = [dconst(0.0) for _ in range(forest.n_classes)]
    for func in tree_funcs:
        tree_res = builder.call(func.llvm_function, args)
        results[func.class_id] = builder.fadd(tree_res, results[func.class_id])
    res_idx = builder.mul(lconst(forest.n_classes), loop_iter_reg)
    results_ptr = [
        builder.gep(out_arr, (builder.add(res_idx, lconst(class_idx)),))
        for class_idx in range(forest.n_classes)
    ]

    results = [
        builder.fadd(result, builder.load(result_ptr))
        for result, result_ptr in zip(results, results_ptr)
    ]
    if eval_obj_func:
        results = _populate_objective_func_block(
            builder,
            results,
            forest.objective_func,
            forest.objective_func_config,
            forest.raw_score,
        )
    for result, result_ptr in zip(results, results_ptr):
        builder.store(result, result_ptr)

    builder.store(builder.add(loop_iter_reg, lconst(1)), loop_iter)
    builder.branch(condition_block)
    # -- END CORE LOOP BLOCK


def _populate_forest_func(forest, root_func, tree_funcs, fblocksize):
    """Populate root function IR for forest"""

    assert fblocksize > 0
    # generate the setup-blocks upfront, so each instruction_block can be passed its successor
    instr_blocks = [
        (
            root_func.append_basic_block("instr-block-setup"),
            tree_funcs[i : i + fblocksize],
        )
        for i in range(0, len(tree_funcs), fblocksize)
    ]
    term_block = root_func.append_basic_block("term")
    ir.IRBuilder(term_block).ret_void()
    for i, (setup_block, tree_func_chunk) in enumerate(instr_blocks):
        next_block = instr_blocks[i + 1][0] if i < len(instr_blocks) - 1 else term_block
        eval_objective_func = next_block == term_block
        _populate_instruction_block(
            forest,
            root_func,
            tree_func_chunk,
            setup_block,
            next_block,
            eval_objective_func,
        )


def _populate_objective_func_block(
    builder, args, objective: str, objective_config: str, raw_score: bool
):
    """
    Takes the objective function specification and generates the code for it into the builder
    """
    llvm_exp = builder.module.declare_intrinsic("llvm.exp", (DOUBLE,))
    llvm_log = builder.module.declare_intrinsic("llvm.log", (DOUBLE,))
    llvm_copysign = builder.module.declare_intrinsic(
        "llvm.copysign", (DOUBLE, DOUBLE), ir.FunctionType(DOUBLE, (DOUBLE, DOUBLE))
    )

    def _populate_sigmoid(alpha):
        if alpha <= 0:
            raise ValueError(f"Sigmoid parameter needs to be >0, is {alpha}")

        # 1 / (1 + exp(- alpha * x))
        inner = builder.fmul(dconst(-alpha), args[0])
        exp = builder.call(llvm_exp, [inner])
        denom = builder.fadd(dconst(1.0), exp)
        return builder.fdiv(dconst(1.0), denom)

    # raw score means we don't need to add the objective function
    if raw_score:
        return args

    if objective == "binary":
        alpha = objective_config.split(":")[1]
        result = _populate_sigmoid(float(alpha))
    elif objective in ("xentropy", "cross_entropy"):
        result = _populate_sigmoid(1.0)
    elif objective in ("xentlambda", "cross_entropy_lambda"):
        # naive implementation which will be numerically unstable for small x.
        # should be changed to log1p
        exp = builder.call(llvm_exp, [args[0]])
        result = builder.call(llvm_log, [builder.fadd(dconst(1.0), exp)])
    elif objective in ("poisson", "gamma", "tweedie"):
        result = builder.call(llvm_exp, [args[0]])
    elif objective in (
        "regression",
        "regression_l1",
        "huber",
        "fair",
        "quantile",
        "mape",
    ):
        if objective_config and "sqrt" in objective_config:
            arg = args[0]
            result = builder.call(llvm_copysign, [builder.fmul(arg, arg), arg])
        else:
            result = args[0]
    elif objective in ("lambdarank", "rank_xendcg", "custom"):
        result = args[0]
    elif objective == "multiclass":
        assert len(args)
        # TODO Might profit from vectorization, needs testing
        result = [builder.call(llvm_exp, [arg]) for arg in args]

        denominator = dconst(0.0)
        for r in result:
            denominator = builder.fadd(r, denominator)

        result = [builder.fdiv(r, denominator) for r in result]
    else:
        raise ValueError(
            f"Objective '{objective}' not yet implemented. {ISSUE_ERROR_MSG}"
        )
    return result if len(args) > 1 else [result]


def _populate_categorical_node_block(
    func, builder, bitset_comp_builder, node, bitset_comp_block, right_block
):
    """Populate block with IR for categorical node"""
    val = func.args[node.split_feature]

    # For categoricals, processing NaNs happens in the Forest root, by explicitly checking for them
    # NaNs are converted to negative max_val, which never exists in the Bitset, so they always go right

    # Find in bitset
    # First, check value > max categorical
    comp = builder.icmp_unsigned(
        "<",
        val,
        iconst(32 * len(node.cat_threshold)),
    )
    builder.cbranch(comp, bitset_comp_block, right_block)

    idx = bitset_comp_builder.udiv(val, iconst(32))
    bit_vecs = ir.Constant(
        ir.VectorType(INT, len(node.cat_threshold)),
        [ir.Constant(INT, i) for i in node.cat_threshold],
    )
    shift = bitset_comp_builder.urem(val, iconst(32))
    # pick relevant bitvector
    bit_vec = bitset_comp_builder.extract_element(bit_vecs, idx)
    # check bitvector contains
    bit_entry = bitset_comp_builder.lshr(bit_vec, shift)
    comp = bitset_comp_builder.trunc(bit_entry, BOOL)
    return comp


def _populate_numerical_node_block(func, builder, node):
    """populate block with IR for numerical node"""
    val = func.args[node.split_feature]

    thresh = ir.Constant(DOUBLE, node.threshold)
    missing_t = node.decision_type.missing_type

    # If missingType != MNaN, LightGBM treats NaNs values as if they were 0.0.
    # So for MZero, NaNs get treated like missing values.
    # But for MNone, NaNs get treated as the literal value 0.0.
    # default_left decides where to go when a missing value is encountered
    # for MNone handle NaNs by adjusting default_left to make sure NaNs go where 0.0 would have gone.
    # for MZero we handle NaNs in the IR
    if node.decision_type.missing_type == MissingType.MNone:
        default_left = 0.0 <= node.threshold
    else:
        default_left = node.decision_type.is_default_left

    # MissingType.MZero: Treat 0s (and NaNs) as missing values
    if default_left:
        if missing_t != MissingType.MZero or (
            missing_t == MissingType.MZero and 0.0 <= node.threshold
        ):
            # unordered cmp: we'll get True (and go left) if any arg is qNaN
            comp = builder.fcmp_unordered("<=", val, thresh)
        else:
            is_missing = builder.fcmp_unordered("==", val, fconst(0.0))
            less_eq = builder.fcmp_unordered("<=", val, thresh)
            comp = builder.or_(is_missing, less_eq)
    else:
        if missing_t != MissingType.MZero or (
            missing_t == MissingType.MZero and node.threshold < 0.0
        ):
            # ordered cmp: we'll get False (and go right) if any arg is qNaN
            comp = builder.fcmp_ordered("<=", val, thresh)
        else:
            is_missing = builder.fcmp_unordered("==", val, fconst(0.0))
            greater = builder.fcmp_ordered(">", val, thresh)
            comp = builder.not_(builder.or_(is_missing, greater))
    return comp
