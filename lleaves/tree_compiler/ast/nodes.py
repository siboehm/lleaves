from llvmlite import ir

from lleaves.tree_compiler.utils import (
    calc_pymode_cat_thresholds,
    decision_idx_to_llvmlite_str,
)

BOOL = ir.IntType(bits=1)
ZERO_V = ir.Constant(BOOL, 0)
DOUBLE = ir.DoubleType()
INT_CAT = ir.IntType(bits=32)


def scalar_func(cat_bitmap):
    return ir.FunctionType(
        DOUBLE, (INT_CAT if is_cat else DOUBLE for is_cat in cat_bitmap)
    )


class Forest:
    """
    Basic outline of the IR representation:

    Forest is a function, which calls the Tree functions

    Trees are functions
    Every node in the tree is a block
    """

    def __init__(self, trees, categorical_bitmap):
        self.trees = trees
        self.categorical_bitmap = categorical_bitmap

    def get_ir(self):
        module = ir.Module(name="forest")

        tree_funcs = [tree.gen_code(module) for tree in self.trees]

        # entry function, do not change name
        root_func = ir.Function(
            module, scalar_func(self.categorical_bitmap), name="forest_root"
        )
        block = root_func.append_basic_block()
        builder = ir.IRBuilder(block)

        res = builder.call(tree_funcs[0], root_func.args)
        for func in tree_funcs[1:]:
            # should probably inline this, but optimizer does it automatically
            tmp = builder.call(func, root_func.args)
            res = builder.fadd(tmp, res)
        builder.ret(res)

        return module

    def _run_pymode(self, input):
        return sum(tree._run_pymode(input) for tree in self.trees)


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
    # the threshold in array-representation
    cat_threshold_arr = None

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

    def finalize_categorical(self, cat_threshold):
        self.cat_threshold = cat_threshold
        self.cat_threshold_arr = calc_pymode_cat_thresholds(cat_threshold)

    def validate(self):
        if self.decision_type_id == 1:
            assert self.cat_threshold is not None
        else:
            assert self.threshold

    def gen_block(self, func):
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        args = func.args

        # numerical float compare
        if self.decision_type_id == 2:
            thresh = ir.Constant(DOUBLE, self.threshold)
            decision_type = decision_idx_to_llvmlite_str(self.decision_type_id)
            comp = builder.fcmp_ordered(decision_type, args[self.split_feature], thresh)
        # categorical int compare
        else:
            decision_type = decision_idx_to_llvmlite_str(self.decision_type_id)
            acc = ZERO_V
            thresholds = [
                ir.Constant(INT_CAT, thresh) for thresh in self.cat_threshold_arr
            ]
            for idx, threshold in enumerate(thresholds):
                tmp = builder.icmp_unsigned(
                    decision_type, args[self.split_feature], threshold
                )
                acc = builder.or_(tmp, acc)
            comp = acc

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
            go_left = input[self.split_feature] in self.cat_threshold_arr

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
