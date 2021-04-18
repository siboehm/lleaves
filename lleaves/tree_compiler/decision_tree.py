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
        return [tree.gen_code() for tree in self.trees]

    def _run_pymode(self, input):
        return sum(tree._run_pymode(input) for tree in self.trees)


class Tree:
    def __init__(self, json, n_args):
        self.index = json["tree_index"]

        # Create an empty module...
        module = ir.Module(name=f"tree_{self.index}")
        # Declare the root node
        func = ir.Function(module, scalar_func(n_args), name=str(self) + "_pred")

        self.func = func
        self.root_node = Node(json["tree_structure"], n_args)

    def __str__(self):
        return f"t_{self.index}"

    def gen_code(self):
        return self.root_node.gen_code(self.func)

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

    def gen_code(self, func, res):
        block = func.append_basic_block(name=str(self))
        builder = ir.IRBuilder(block)
        args = func.args

        thresh = ir.Constant(DOUBLE, self.threshold)
        comp = builder.fcmp_ordered(
            self.decision_type, args[self.split_feature], thresh, name=str(self) + "_c"
        )
        with builder.if_else(comp) as (left, right):
            with left:
                builder.branch(self.left.gen_code(func, res))
            with right:
                builder.branch(self.right.gen_code(func, res))
        result = builder.load(res)
        builder.ret(res)

    def __str__(self):
        return f"n_{self.node_index}"

    def _run_pymode(self, input):
        if input[self.split_feature] <= self.threshold:
            return self.left._run_pymode(input)
        else:
            return self.right._run_pymode(input)


class Leaf:
    def __init__(self, json):
        self.return_value = json["leaf_value"]

    def gen_code(self):
        pass

    def _run_pymode(self, input):
        return self.return_value
