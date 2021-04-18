from llvmlite import ir

# Create some useful types
DOUBLE = ir.DoubleType()
D_POINTER = ir.PointerType(DOUBLE)


def scalar_func(n_args):
    return ir.FunctionType(DOUBLE, n_args * (DOUBLE,))


class Forest:
    def __init__(self, json):
        self.trees = [
            Tree(tree_json, len(json["feature_infos"]))
            for tree_json in json["tree_info"]
        ]

    def get_ir(self):
        return [str(tree.gen_code()) for tree in self.trees]

    def _run_pymode(self, input):
        return sum(tree._run_pymode(input) for tree in self.trees)


class Tree:
    def __init__(self, json, n_args):
        self.index = json["tree_index"]
        self.n_args = n_args

        self.root_node = Node(json["tree_structure"], n_args)

    def __str__(self):
        return f"tree_{self.index}"

    def gen_code(self):
        # Create an empty module...
        module = ir.Module(name=f"tree_{self.index}")
        # Declare the function for this tree
        func = ir.Function(module, scalar_func(self.n_args), name=str(self) + "_pred")
        block = func.append_basic_block("entry")

        builder = ir.IRBuilder(block)
        res = builder.alloca(DOUBLE, name="result")
        builder.branch(self.root_node.gen_block(func, res))

        return module

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

        self.node_index = json["split_index"]
        self.split_feature = json["split_feature"]
        self.threshold = json["threshold"]
        self.decision_type = json["decision_type"]

    def gen_block(self, func, res):
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        args = func.args

        thresh = ir.Constant(DOUBLE, self.threshold)
        comp = builder.fcmp_ordered(
            self.decision_type, args[self.split_feature], thresh
        )
        with builder.if_then(comp):
            self._branch_or_ret(self.left, builder, func, res)
        self._branch_or_ret(self.right, builder, func, res)
        return block

    @staticmethod
    def _branch_or_ret(target, builder, func, res):
        if isinstance(target, Leaf):
            target.add_return(builder)
        else:
            builder.branch(target.gen_block(func, res))

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

    def add_return(self, builder):
        result = ir.Constant(DOUBLE, self.return_value)
        builder.ret(result)

    def _run_pymode(self, input):
        return self.return_value

    def __str__(self):
        return f"leaf_{self.leaf_index}"
