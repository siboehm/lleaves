from lleaves.compiler.ast.nodes import DecisionNode, Forest, LeafNode, Tree
from lleaves.compiler.ast.scanner import cat_args_bitmap, scan_model_file
from lleaves.compiler.utils import DecisionType


def _parse_tree_to_ast(tree_struct, cat_bitmap):
    n_nodes = len(tree_struct["decision_type"])
    leaves = [
        LeafNode(idx, value) for idx, value in enumerate(tree_struct["leaf_value"])
    ]

    # Create the nodes using all non-specific data
    # categorical nodes are finalized later
    nodes = [
        DecisionNode(
            idx, split_feature, threshold, decision_type_id, left_idx, right_idx
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
        return Tree(tree_struct["Tree"], nodes[0], cat_bitmap)
    else:
        # special case for when tree is just single leaf
        assert len(leaves) == 1
        return Tree(tree_struct["Tree"], leaves[0], cat_bitmap)


def parse_to_ast(model_path):
    scanned_model = scan_model_file(model_path)
    n_args = scanned_model["general_info"]["max_feature_idx"] + 1
    cat_bitmap = cat_args_bitmap(scanned_model["general_info"]["feature_infos"])
    assert n_args == len(cat_bitmap), "Ill formed model file"

    trees = [
        _parse_tree_to_ast(tree_struct, cat_bitmap)
        for tree_struct in scanned_model["trees"]
    ]
    return Forest(trees, cat_bitmap)
