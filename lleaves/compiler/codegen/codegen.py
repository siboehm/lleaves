from llvmlite import ir

from lleaves.compiler.utils import MissingType

BOOL = ir.IntType(bits=1)
DOUBLE = ir.DoubleType()
FLOAT = ir.FloatType()
INT_CAT = ir.IntType(bits=32)
INT = ir.IntType(bits=32)
ZERO_V = ir.Constant(BOOL, 0)
FLOAT_POINTER = ir.PointerType(FLOAT)
DOUBLE_PTR = ir.PointerType(DOUBLE)


def iconst(value):
    return ir.Constant(INT, value)


def fconst(value):
    return ir.Constant(FLOAT, value)


def dconst(value):
    return ir.Constant(DOUBLE, value)


def scalar_func(cat_bitmap):
    return ir.FunctionType(
        DOUBLE, (INT_CAT if is_cat else DOUBLE for is_cat in cat_bitmap)
    )


def ir_from_ast(forest):
    module = ir.Module(name="forest")

    # entry function called from Python
    root_func = ir.Function(
        module,
        ir.FunctionType(ir.VoidType(), (DOUBLE_PTR, DOUBLE_PTR, INT, INT)),
        name="forest_root",
    )

    tree_funcs = []
    for tree in forest.trees:
        # Declare the function for this tree
        tree_func = ir.Function(
            module, scalar_func(tree.categorical_bitmap), name=str(tree)
        )
        # add IR
        gen_tree(tree, tree_func)
        tree_funcs.append(tree_func)
    populate_forest_func(forest, root_func, tree_funcs)

    return module


def gen_tree(tree, tree_func):
    """generate code for tree given the function, recursing into nodes"""
    node_block = tree_func.append_basic_block(name=str(tree.root_node))
    gen_node(tree_func, node_block, tree.root_node)


def gen_node(func, node_block, node):
    """generate code for node, recursing into children"""
    if node.is_leaf():
        gen_leaf_node(node_block, node)
    else:
        gen_decision_node(func, node_block, node)


def gen_leaf_node(node_block, leaf):
    """populate block with leaf's return value"""
    builder = ir.IRBuilder(node_block)
    builder.ret(dconst(leaf.value))


def gen_decision_node(func, node_block, node):
    """generate code for decision node, recursing into children"""
    builder = ir.IRBuilder(node_block)

    # optimization for node where both children are leaves (switch instead of cbranch)
    is_fused_double_leaf_node = node.left.is_leaf() and node.right.is_leaf()
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

    # If missingType != MNaN, LightGBM treats NaNs values as if they were 0.0.
    # So for MZero, NaNs get treated like missing values.
    # But for MNone, NaNs get treated as the literal value 0.0.
    # default_left decides where to go when a missing value is encountered

    if node.decision_type.is_categorical:
        bitset_comp_block = builder.append_basic_block("cat_bitset_comp")
        bitset_builder = ir.IRBuilder(bitset_comp_block)
        comp = populate_categorical_node_block(
            func, builder, bitset_builder, node, bitset_comp_block, right_block
        )
        builder = bitset_builder
    else:
        comp = populate_numerical_node_block(func, builder, node)

    if is_fused_double_leaf_node:
        ret = builder.select(comp, dconst(node.left.value), dconst(node.right.value))
        builder.ret(ret)
    else:
        builder.cbranch(comp, left_block, right_block)

    if left_block:
        gen_node(func, left_block, node.left)
    if right_block:
        gen_node(func, right_block, node.right)


def populate_forest_func(forest, root_func, tree_funcs):
    """Populate root function IR for forest"""
    data_arr, out_arr, start_index, end_index = root_func.args

    # -- SETUP BLOCK
    setup_block = root_func.append_basic_block("setup")
    builder = ir.IRBuilder(setup_block)
    loop_iter = builder.alloca(INT, 1, "loop-idx")
    builder.store(start_index, loop_iter)
    condition_block = root_func.append_basic_block("loop-condition")
    builder.branch(condition_block)
    # -- END SETUP BLOCK

    # -- CONDITION BLOCK
    builder = ir.IRBuilder(condition_block)
    comp = builder.icmp_signed("<", builder.load(loop_iter), end_index)
    core_block = root_func.append_basic_block("loop-core")
    term_block = root_func.append_basic_block("term")
    builder.cbranch(comp, core_block, term_block)
    # -- END CONDITION BLOCK

    # -- CORE LOOP BLOCK
    builder = ir.IRBuilder(core_block)
    # build args arr, convert categoricals vars from float to int
    args = []
    loop_iter_reg = builder.load(loop_iter)

    n_args = ir.Constant(INT, forest.n_args)
    iter_mul_nargs = builder.mul(loop_iter_reg, n_args)
    idx = (builder.add(iter_mul_nargs, iconst(i)) for i in range(forest.n_args))
    raw_ptrs = [builder.gep(root_func.args[0], (c,)) for c in idx]
    for is_cat, ptr in zip(forest.categorical_bitmap, raw_ptrs):
        el = builder.load(ptr)
        if is_cat:
            args.append(builder.fptosi(el, INT_CAT))
        else:
            args.append(el)
    # iterate over each tree, sum up results
    res = builder.call(tree_funcs[0], args)
    for func in tree_funcs[1:]:
        # could be inlined, but optimizer does for us
        tree_res = builder.call(func, args)
        res = builder.fadd(tree_res, res)
    ptr = builder.gep(out_arr, (loop_iter_reg,))
    builder.store(res, ptr)
    tmpp1 = builder.add(loop_iter_reg, iconst(1))
    builder.store(tmpp1, loop_iter)
    builder.branch(condition_block)
    # -- END CORE LOOP BLOCK

    # -- TERMINAL BLOCK
    ir.IRBuilder(term_block).ret_void()
    # -- END TERMINAL BLOCK


def populate_categorical_node_block(
    func, builder, bitset_comp_builder, node, bitset_comp_block, right_block
):
    """Populate block with IR for categorical node"""
    val = func.args[node.split_feature]

    # For categoricals, processing NaNs happens through casting them via fptosi in the Forest root
    # NaNs become negative max_val, which never exists in the Bitset, so they always go right

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


def populate_numerical_node_block(func, builder, node):
    """populate block with IR for numerical node"""
    val = func.args[node.split_feature]

    thresh = ir.Constant(DOUBLE, node.threshold)
    missing_t = node.decision_type.missing_type

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
