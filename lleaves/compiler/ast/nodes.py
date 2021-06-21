from lleaves.compiler.utils import DecisionType


class Forest:
    def __init__(self, trees: list, categorical_bitmap: list):
        """
        :param trees: list of trees
        :param categorical_bitmap: bitmap indicating categorical / numerical feature
        """
        self.trees = trees
        self.categorical_bitmap = categorical_bitmap
        self.n_args = len(categorical_bitmap)


class Tree:
    def __init__(self, idx, root_node, categorical_bitmap):
        self.idx = idx
        self.root_node = root_node
        self.categorical_bitmap = categorical_bitmap

    def __str__(self):
        return f"tree_{self.idx}"


class Node:
    def is_leaf(self):
        return isinstance(self, LeafNode)


class DecisionNode(Node):
    # the threshold in bit-representation if this node is categorical
    cat_threshold = None

    # child nodes
    left = None
    right = None

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

    def __str__(self):
        return f"node_{self.idx}"


class LeafNode(Node):
    def __init__(self, idx, value):
        self.idx = idx
        self.value = value

    def __str__(self):
        return f"leaf_{self.idx}"
