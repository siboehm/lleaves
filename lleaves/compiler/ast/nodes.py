from dataclasses import dataclass, field
from typing import List

from lleaves.compiler.utils import DecisionType


class Node:
    @property
    def is_leaf(self):
        return isinstance(self, LeafNode)


@dataclass
class Tree:
    idx: int
    root_node: Node
    features: list
    class_id: int

    def __str__(self):
        return f"tree_{self.idx}"


@dataclass
class Forest:
    trees: List[Tree]
    features: list
    n_classes: int
    objective_func: str
    objective_func_config: str
    raw_score: bool = False

    @property
    def n_args(self):
        return len(self.features)


@dataclass
class DecisionNode(Node):
    # the threshold in bit-representation if this node is categorical
    cat_threshold: List[int] = field(default=None, init=False)

    # child nodes
    left: Node = field(default=None, init=False)
    right: Node = field(default=None, init=False)

    idx: int
    split_feature: int
    threshold: int
    decision_type: DecisionType
    left_idx: int
    right_idx: int

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


@dataclass
class LeafNode(Node):
    idx: int
    value: float

    def __str__(self):
        return f"leaf_{self.idx}"
