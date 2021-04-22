from lleaves.tree_compiler.frontend import Forest, Leaf, Node, Tree
from lleaves.tree_compiler.parser import parse_model_file


def parse_to_forest(model_path):
    parsed_model = parse_model_file(model_path)
    n_args = parsed_model["general_info"]["max_feature_idx"] + 1

    trees = []
    for tree_struct in parsed_model["trees"]:
        leaves = [
            Leaf(idx, value) for idx, value in enumerate(tree_struct["leaf_value"])
        ]
        nodes = [
            Node(idx, split_feature, threshold, decision_type_id, left_idx, right_idx)
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

        for node in nodes:
            # in the model_file.txt, the outgoing left + right nodes are specified
            # via their index in the list. negative numbers are leaves, positive numbers
            # are other nodes
            children = [
                leaves[abs(idx) - 1] if idx < 0 else nodes[idx]
                for idx in (node.left_idx, node.right_idx)
            ]
            node.add_children(*children)
        trees.append(Tree(tree_struct["Tree"], nodes[0], n_args))
    return Forest(trees, n_args)
