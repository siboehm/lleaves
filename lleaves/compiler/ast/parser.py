import itertools

from lleaves.compiler.ast.nodes import DecisionNode, Forest, LeafNode, Tree
from lleaves.compiler.ast.scanner import scan_model_file
from lleaves.compiler.utils import DecisionType

"""
The parser takes the results from the scanner and transforms them into an
Abstract-Syntax Tree (AST).
It builds up the Graph for the DecisionTree-Forest, consisting of decision-nodes and leaf-nodes.
"""


class Feature:
    """
    Represents one feature (= column) that is passed to the tree function or forest function.
    """

    def __init__(self, is_categorical):
        self.is_categorical = is_categorical


def _parse_tree_to_ast(tree_struct, features, class_id):
    n_nodes = len(tree_struct["decision_type"])
    leaves = [
        LeafNode(idx, value) for idx, value in enumerate(tree_struct["leaf_value"])
    ]

    # Create the nodes using all non-specific data
    # categorical nodes are finalized later
    nodes = [
        DecisionNode(
            idx,
            split_feature,
            threshold,
            DecisionType(decision_type_id),
            left_idx,
            right_idx,
        )
        for idx, (
            split_feature,
            threshold,
            decision_type_id,
            left_idx,
            right_idx,
        ) in enumerate(
            zip(
                tree_struct["split_feature"],
                tree_struct["threshold"],
                tree_struct["decision_type"],
                tree_struct["left_child"],
                tree_struct["right_child"],
            )
        )
    ]
    assert len(nodes) == n_nodes

    categorical_nodes = [
        idx
        for idx, decision_type_id in enumerate(tree_struct["decision_type"])
        if DecisionType(decision_type_id).is_categorical
    ]

    for idx in categorical_nodes:
        node = nodes[idx]
        thresh = int(node.threshold)
        # pass just the relevant vector entries
        start = tree_struct["cat_boundaries"][thresh]
        end = tree_struct["cat_boundaries"][thresh + 1]
        node.finalize_categorical(
            cat_threshold=tree_struct["cat_threshold"][start:end],
        )

    for node in nodes:
        # in the model_file.txt, the outgoing left + right nodes are specified
        # via their index in the list. negative numbers are leaves, positive numbers
        # are other nodes
        children = [
            leaves[abs(idx) - 1] if idx < 0 else nodes[idx]
            for idx in (node.left_idx, node.right_idx)
        ]
        node.add_children(*children)

    for node in nodes:
        node.validate()

    if nodes:
        return Tree(tree_struct["Tree"], nodes[0], features, class_id)
    else:
        # special case for when tree is just single leaf
        assert len(leaves) == 1
        return Tree(tree_struct["Tree"], leaves[0], features, class_id)


def parse_to_ast(model_path):
    scanned_model = scan_model_file(model_path)

    n_args = scanned_model["general_info"]["max_feature_idx"] + 1
    n_classes = scanned_model["general_info"]["num_class"]
    assert n_classes == scanned_model["general_info"]["num_tree_per_iteration"]
    objective = scanned_model["general_info"]["objective"]
    objective_func = objective[0]
    objective_func_config = objective[1] if len(objective) > 1 else None
    features = [
        Feature(is_categorical_feature(x))
        for x in scanned_model["general_info"]["feature_infos"]
    ]
    assert n_args == len(features), "Ill formed model file"

    trees = [
        _parse_tree_to_ast(scanned_tree, features, class_id)
        for scanned_tree, class_id in zip(
            scanned_model["trees"], itertools.cycle(range(n_classes))
        )
    ]
    assert len(trees) % n_classes == 0, "Ill formed model file"
    return Forest(trees, features, n_classes, objective_func, objective_func_config)


def is_categorical_feature(feature_info: str):
    """
    :param feature_info: one entry from the model.txt 'feature_infos' field
    """
    # Feature infos for floats look like [x.xxxx:y.yyyy]
    # for categoricals like X:Y:Z:
    return not feature_info.startswith("[")
