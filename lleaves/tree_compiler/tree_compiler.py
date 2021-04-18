from llvmlite import ir

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

    def __init__(self, json):
        self.n_args = len(json["feature_infos"])
        self.trees = [Tree(tree_json, self.n_args) for tree_json in json["tree_info"]]

    def get_ir(self):
        module = ir.Module(name=f"forest")

        tree_funcs = [tree.gen_code(module) for tree in self.trees]

        # entry function, do not change name
        root_func = ir.Function(module, scalar_func(self.n_args), name="forest_root")
        block = root_func.append_basic_block()
        builder = ir.IRBuilder(block)

        res = builder.call(tree_funcs[0], root_func.args)
        for func in tree_funcs[1:]:
            tmp = builder.call(func, root_func.args)
            res = builder.fadd(tmp, res)
        builder.ret(res)

        return module

    def _run_pymode(self, input):
        return sum(tree._run_pymode(input) for tree in self.trees)


class Tree:
    def __init__(self, json, n_args):
        self.index = json["tree_index"]
        self.n_args = n_args

        self.root_node = Node(json["tree_structure"], n_args)

    def __str__(self):
        return f"tree_{self.index}"

    def gen_code(self, module):
        # Declare the function for this tree
        func = ir.Function(module, scalar_func(self.n_args), name=str(self))
        # root node will provide the first block, which is the entry into the tree function
        self.root_node.gen_block(func)
        return func

    def _run_pymode(self, input):
        return self.root_node._run_pymode(input)


class Node:
    right = None
    left = None

    def __init__(self, json, n_args):
        if "split_feature" in json["left_child"]:
            self.left = Node(json["left_child"], n_args)
        else:
            self.left = Leaf(json["left_child"])

        if "split_feature" in json["right_child"]:
            self.right = Node(json["right_child"], n_args)
        else:
            self.right = Leaf(json["right_child"])

        self.all_children_leaves = isinstance(self.left, Leaf) and isinstance(self.right, Leaf)

        self.node_index = json["split_index"]
        self.split_feature = json["split_feature"]
        self.threshold = json["threshold"]
        self.decision_type = json["decision_type"]

    def gen_block(self, func):
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        args = func.args

        thresh = ir.Constant(DOUBLE, self.threshold)
        comp = builder.fcmp_ordered(
            self.decision_type, args[self.split_feature], thresh
        )
        if self.all_children_leaves:
            ret = builder.select(comp, self.left.return_const, self.right.return_const)
            builder.ret(ret)
        else:
            builder.cbranch(comp, self.left.gen_block(func), self.right.gen_block(func))

        return block

    def __str__(self):
        return f"node_{self.node_index}"

    def _run_pymode(self, input):
        if input[self.split_feature] <= self.threshold:
            return self.left._run_pymode(input)
        else:
            return self.right._run_pymode(input)


class Leaf:
    def __init__(self, json):
        self.leaf_index = json["leaf_index"]
        self.return_value = json["leaf_value"]
        self.return_const = ir.Constant(DOUBLE, json["leaf_value"])

    def gen_block(self, func):
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        builder.ret(self.return_const)
        return block

    def _run_pymode(self, input):
        return self.return_value

    def __str__(self):
        return f"leaf_{self.leaf_index}"
