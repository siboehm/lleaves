from llvmlite import ir
from lleaves.tree_compiler.utils import decision_idx_to_llvmlite_str

DOUBLE = ir.DoubleType()


def scalar_func(n_args):
    return ir.FunctionType(DOUBLE, n_args * (DOUBLE,))


class Forest:
    """
    Basic outline of the IR representation:

    Forest is a function, which calls the Tree functions

    Trees are functions
    Every node in the tree is a block
    """

    def __init__(self, trees, n_args):
        self.trees = trees
        self.n_args = n_args

    def get_ir(self):
        module = ir.Module(name=f"forest")

        tree_funcs = [tree.gen_code(module) for tree in self.trees]

        # entry function, do not change name
        root_func = ir.Function(module, scalar_func(self.n_args), name="forest_root")
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
    def __init__(self, idx, root_node, n_args):
        self.idx = idx
        self.root_node = root_node
        self.n_args = n_args

    def __str__(self):
        return f"tree_{self.idx}"

    def gen_code(self, module):
        # Declare the function for this tree
        func = ir.Function(module, scalar_func(self.n_args), name=str(self))
        # root node will provide the first block, which is the entry into the tree function
        self.root_node.gen_block(func)
        return func

    def _run_pymode(self, input):
        return self.root_node._run_pymode(input)


class Node:
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

    def gen_block(self, func):
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        args = func.args

        thresh = ir.Constant(DOUBLE, self.threshold)
        decision_type = decision_idx_to_llvmlite_str(self.decision_type_id)
        comp = builder.fcmp_ordered(decision_type, args[self.split_feature], thresh)
        if self.all_children_leaves:
            ret = builder.select(comp, self.left.return_const, self.right.return_const)
            builder.ret(ret)
        else:
            builder.cbranch(comp, self.left.gen_block(func), self.right.gen_block(func))

        return block

    def __str__(self):
        return f"node_{self.idx}"

    def _run_pymode(self, input):
        if input[self.split_feature] <= self.threshold:
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
