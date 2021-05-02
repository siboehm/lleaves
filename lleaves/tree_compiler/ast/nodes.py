from llvmlite import ir

from lleaves.tree_compiler.utils import (
    calc_pymode_cat_thresholds,
    decision_idx_to_llvmlite_str,
)

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
        idx = [
            builder.add(iter_mul_nargs, c)
            for c in (ir.Constant(INT, i) for i in range(self.n_args))
        ]
        raw_ptrs = [builder.gep(root_func.args[0], (c,)) for c in idx]
        for is_cat, ptr in zip(self.categorical_bitmap, raw_ptrs):
            el = builder.load(ptr)
            if is_cat:
                args.append(builder.fptoui(el, INT_CAT))
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

    def _run_pymode(self, inputs):
        return [sum(tree._run_pymode(input) for tree in self.trees) for input in inputs]


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

    def _run_pymode(self, input):
        return self.root_node._run_pymode(input)


class Node:
    # the threshold in bit-representation if this node is categorical
    cat_threshold = None
    # The boundary conditions if this node is categorical
    cat_boundary = None
    cat_boundary_pp = None

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
        self.decision_type_id = decision_type_id
        self.right_idx = right_idx
        self.left_idx = left_idx

    def add_children(self, left, right):
        self.left = left
        self.right = right

        self.all_children_leaves = isinstance(self.left, Leaf) and isinstance(
            self.right, Leaf
        )

    def finalize_categorical(self, cat_threshold, cat_boundary, cat_boundary_pp):
        self.cat_threshold = cat_threshold
        self.cat_boundary = cat_boundary
        self.cat_boundary_pp = cat_boundary_pp
        self.threshold = int(self.threshold)

    def validate(self):
        if self.decision_type_id == 1:
            assert self.cat_threshold is not None
        else:
            assert self.threshold

    def gen_block(self, func):
        print("dkdkd")
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        args = func.args

        # numerical float compare
        decision_type = decision_idx_to_llvmlite_str(self.decision_type_id)
        if decision_type == "<=":
            thresh = ir.Constant(DOUBLE, self.threshold)
            decision_type = decision_idx_to_llvmlite_str(self.decision_type_id)
            comp = builder.fcmp_ordered(decision_type, args[self.split_feature], thresh)
        # categorical int compare
        else:
            # find in bitset
            # check > max
            i1 = builder.sdiv(args[self.split_feature], ir.Constant(INT, 32))
            comp1 = builder.icmp_signed(
                "<", i1, ir.Constant(INT, self.cat_boundary_pp - self.cat_boundary)
            )
            # check arg contained in bitvector
            bit_entries = self.cat_threshold[self.cat_boundary :]
            bit_vecs = ir.Constant(
                ir.VectorType(INT, len(bit_entries)),
                [ir.Constant(INT, i) for i in bit_entries],
            )
            shift = builder.srem(args[self.split_feature], ir.Constant(INT, 32))
            bit_vec = builder.extract_element(bit_vecs, i1)
            bit_entry = builder.lshr(bit_vec, shift)
            bit_val = builder.and_(bit_entry, ir.Constant(INT, 1))
            comp2 = builder.icmp_signed("==", bit_val, ir.Constant(INT, 1))
            comp = builder.and_(comp1, comp2)

        if self.all_children_leaves:
            ret = builder.select(comp, self.left.return_const, self.right.return_const)
            builder.ret(ret)
        else:
            builder.cbranch(comp, self.left.gen_block(func), self.right.gen_block(func))

        return block

    def __str__(self):
        return f"node_{self.idx}"

    def _run_pymode(self, input):
        if self.decision_type_id == 2:
            go_left = input[self.split_feature] <= self.threshold
        else:
            go_left = input[self.split_feature] in calc_pymode_cat_thresholds(
                self.cat_threshold[self.cat_boundary]
            )

        if go_left:
            return self.left._run_pymode(input)
        else:
            return self.right._run_pymode(input)


class Leaf:
    def __init__(self, idx, value):
        self.idx = idx
        self.value = value
        self.return_const = ir.Constant(DOUBLE, value)

    def gen_block(self, func):
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        builder.ret(self.return_const)
        return block

    def _run_pymode(self, input):
        return self.value

    def __str__(self):
        return f"leaf_{self.idx}"
