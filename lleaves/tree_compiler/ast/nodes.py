from llvmlite import ir

from lleaves.tree_compiler.utils import DecisionType, MissingType

BOOL = ir.IntType(bits=1)
DOUBLE = ir.DoubleType()
FLOAT = ir.FloatType()
INT_CAT = ir.IntType(bits=32)
INT = ir.IntType(bits=32)
ZERO_V = ir.Constant(BOOL, 0)
FLOAT_POINTER = ir.PointerType(FLOAT)
DOUBLE_PTR = ir.PointerType(DOUBLE)


def scalar_func(cat_bitmap):
    return ir.FunctionType(
        DOUBLE, (INT_CAT if is_cat else DOUBLE for is_cat in cat_bitmap)
    )


class Forest:
    """
    Basic outline of the IR representation:

    Forest is a function, which calls the Tree functions

    Every node in the tree is a block
    """

    def __init__(self, trees, categorical_bitmap):
        self.trees = trees
        self.categorical_bitmap = categorical_bitmap
        self.n_args = len(categorical_bitmap)

    def get_ir(self):
        module = ir.Module(name="forest")

        tree_funcs = [tree.gen_code(module) for tree in self.trees]

        # entry function called from Python via CFUNC
        root_func = ir.Function(
            module,
            ir.FunctionType(ir.VoidType(), (DOUBLE_PTR, INT, DOUBLE_PTR)),
            name="forest_root",
        )

        data_arr, n_pred, out_arr = root_func.args

        # -- SETUP BLOCK
        setup_block = root_func.append_basic_block("setup")
        builder = ir.IRBuilder(setup_block)
        loop_iter = builder.alloca(INT, 1, "loop-idx")
        builder.store(ir.Constant(INT, 0), loop_iter)
        condition_block = root_func.append_basic_block("loop-condition")
        builder.branch(condition_block)
        # -- END SETUP BLOCK

        # -- CONDITION BLOCK
        builder = ir.IRBuilder(condition_block)
        comp = builder.icmp_signed("<", builder.load(loop_iter), n_pred)
        core_block = root_func.append_basic_block("loop-core")
        term_block = root_func.append_basic_block("term")
        builder.cbranch(comp, core_block, term_block)
        # -- END CONDITION BLOCK

        # -- CORE LOOP BLOCK
        builder = ir.IRBuilder(core_block)
        # build args arr, convert categoricals vars from float to int
        args = []
        loop_iter_reg = builder.load(loop_iter)

        n_args = ir.Constant(INT, self.n_args)
        iter_mul_nargs = builder.mul(loop_iter_reg, n_args)
        idx = (
            builder.add(iter_mul_nargs, ir.Constant(INT, i)) for i in range(self.n_args)
        )
        raw_ptrs = [builder.gep(root_func.args[0], (c,)) for c in idx]
        for is_cat, ptr in zip(self.categorical_bitmap, raw_ptrs):
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
        tmpp1 = builder.add(loop_iter_reg, ir.Constant(INT, 1))
        builder.store(tmpp1, loop_iter)
        builder.branch(condition_block)
        # -- END CORE LOOP BLOCK

        # -- TERMINAL BLOCK
        ir.IRBuilder(term_block).ret_void()
        # -- END TERMINAL BLOCK

        return module


class Tree:
    def __init__(self, idx, root_node, categorical_bitmap):
        self.idx = idx
        self.root_node = root_node
        self.categorical_bitmap = categorical_bitmap

    def __str__(self):
        return f"tree_{self.idx}"

    def gen_code(self, module):
        # Declare the function for this tree
        func = ir.Function(module, scalar_func(self.categorical_bitmap), name=str(self))
        # root node will provide the first block, which is the entry into the tree function
        self.root_node.gen_block(func)
        return func


class Node:
    def is_leaf(self):
        return isinstance(self, Leaf)


class InnerNode(Node):
    # the threshold in bit-representation if this node is categorical
    cat_threshold = None

    # child nodes or leaves
    left = None
    right = None

    # the IR block to jump to to access this node
    _node_block = None

    def __init__(
        self,
        idx: int,
        split_feature: int,
        threshold: int,
        decision_type_id: int,
        left_idx: int,
        right_idx: int,
    ):
        self.idx = idx
        self.split_feature = split_feature
        self.threshold = threshold
        self.decision_type = DecisionType(decision_type_id)
        self.right_idx = right_idx
        self.left_idx = left_idx

    def add_children(self, left, right):
        self.left = left
        self.right = right

    def finalize_categorical(self, cat_threshold):
        self.cat_threshold = cat_threshold
        self.threshold = int(self.threshold)

    def validate(self):
        if self.decision_type.is_categorical:
            assert self.cat_threshold is not None
        else:
            assert self.threshold

    def gen_block(self, func):
        if self._node_block:
            return self._node_block

        node_block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(node_block)
        val = func.args[self.split_feature]

        # If missingType != MNaN, LightGBM treats NaNs values as if they were 0.0.
        # So for MZero, NaNs get treated like missing values.
        # But for MNone, NaNs get treated as the literal value 0.0.

        # default_left decides where to go when a missing value is encountered

        # categorical int compare
        if self.decision_type.is_categorical:
            # For categoricals, processing NaNs happens through casting them via fptosi in the Forest root
            # NaNs become negative max_val, which never exists in the Bitset, so they always go right
            # This seems to be the default LightGBM behaviour, but it's hard to tell from their code.

            # Find in bitset
            # First, check value > max categorical
            comp = builder.icmp_unsigned(
                "<",
                val,
                ir.Constant(INT, 32 * len(self.cat_threshold)),
            )
            bitset_comp_block = builder.append_basic_block("cat_bitset_comp")
            builder.cbranch(comp, bitset_comp_block, self.right.gen_block(func))
            builder = ir.IRBuilder(bitset_comp_block)

            idx = builder.udiv(val, ir.Constant(INT, 32))
            bit_vecs = ir.Constant(
                ir.VectorType(INT, len(self.cat_threshold)),
                [ir.Constant(INT, i) for i in self.cat_threshold],
            )
            shift = builder.urem(val, ir.Constant(INT, 32))
            # pick relevant bitvector
            bit_vec = builder.extract_element(bit_vecs, idx)
            # check bitvector contains
            bit_entry = builder.lshr(bit_vec, shift)
            comp = builder.trunc(bit_entry, BOOL)
        # numerical float compare
        else:
            thresh = ir.Constant(DOUBLE, self.threshold)
            missing_t = self.decision_type.missing_type

            # for MNone handle NaNs by adjusting default_left to make sure NaNs go where 0.0 would have gone.
            # for MZero we handle NaNs in the IR
            if self.decision_type.missing_type == MissingType.MNone:
                default_left = 0.0 <= self.threshold
            else:
                default_left = self.decision_type.is_default_left

            # MissingType.MZero: Treat 0s (and NaNs) as missing values
            if default_left:
                if missing_t != MissingType.MZero or (
                    missing_t == MissingType.MZero and 0.0 <= self.threshold
                ):
                    # unordered cmp: we'll get True (and go left) if any arg is qNaN
                    comp = builder.fcmp_unordered("<=", val, thresh)
                else:
                    is_missing = builder.fcmp_unordered(
                        "==", val, ir.Constant(FLOAT, 0.0)
                    )
                    less_eq = builder.fcmp_unordered("<=", val, thresh)
                    comp = builder.or_(is_missing, less_eq)
            else:
                if missing_t != MissingType.MZero or (
                    missing_t == MissingType.MZero and self.threshold < 0.0
                ):
                    # ordered cmp: we'll get False (and go right) if any arg is qNaN
                    comp = builder.fcmp_ordered("<=", val, thresh)
                else:
                    is_missing = builder.fcmp_unordered(
                        "==", val, ir.Constant(FLOAT, 0.0)
                    )
                    greater = builder.fcmp_ordered(">", val, thresh)
                    comp = builder.not_(builder.or_(is_missing, greater))

        if self.right.is_leaf() and self.left.is_leaf():
            ret = builder.select(comp, self.left.return_const, self.right.return_const)
            builder.ret(ret)
        else:
            builder.cbranch(comp, self.left.gen_block(func), self.right.gen_block(func))

        self._node_block = node_block
        return self._node_block

    def __str__(self):
        return f"node_{self.idx}"


class Leaf(Node):
    _leaf_block = None

    def __init__(self, idx, value):
        self.idx = idx
        self.value = value
        self.return_const = ir.Constant(DOUBLE, value)

    def gen_block(self, func):
        if not self._leaf_block:
            block = func.append_basic_block(name=str(self))
            builder = ir.IRBuilder(block)
            builder.ret(self.return_const)
            self._leaf_block = block
        return self._leaf_block

    def __str__(self):
        return f"leaf_{self.idx}"
