from lleaves.tree_compiler.ast.nodes import Forest, Leaf, Node, Tree
from lleaves.tree_compiler.ast.parser import cat_args_bitmap, parse_model_file


def parse_to_ast(model_path):
    parsed_model = parse_model_file(model_path)
    n_args = parsed_model["general_info"]["max_feature_idx"] + 1
    cat_bitmap = cat_args_bitmap(parsed_model["general_info"]["feature_infos"])
    assert n_args == len(cat_bitmap), "Ill formed model file"

    trees = []
    for tree_struct in parsed_model["trees"]:
        n_cat = tree_struct["num_cat"]
        n_nodes = len(tree_struct["decision_type"])
        leaves = [
            Leaf(idx, value) for idx, value in enumerate(tree_struct["leaf_value"])
        ]

        # Create the nodes using all non-specific data
        # categorical nodes are finalized later
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
        assert len(nodes) == n_nodes

        categorical_nodes = [
            idx
            for idx, decision_type in enumerate(tree_struct["decision_type"])
            if decision_type == 1 or decision_type == 9
        ]
        assert len(categorical_nodes) == n_cat

        for idx in categorical_nodes:
            node = nodes[idx]
            thresh = int(node.threshold)
            node.finalize_categorical(
                cat_threshold=tree_struct["cat_threshold"],
                cat_boundary=tree_struct["cat_boundaries"][thresh],
                cat_boundary_pp=tree_struct["cat_boundaries"][thresh + 1],
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

        trees.append(Tree(tree_struct["Tree"], nodes[0], cat_bitmap))
    return Forest(trees, cat_bitmap)
