from lleaves.compiler.ast.scanner import scan_model_file


def test_parser():
    model_file = "tests/models/boston_housing/model.txt"
    result = scan_model_file(model_file)
    assert result["general_info"]["max_feature_idx"] == 12
    assert len(result["trees"]) == 100

    tree_3 = result["trees"][3]
    assert tree_3["num_leaves"] == 18
    assert tree_3["left_child"] == [
        1,
        3,
        -2,
        -1,
        9,
        8,
        7,
        10,
        15,
        16,
        -5,
        -9,
        -8,
        14,
        -14,
        -6,
        -3,
    ]

    tree_95 = result["trees"][95]
    assert tree_95["Tree"] == 95
    assert tree_95["num_leaves"] == 21

    model_file = "tests/models/tiniest_single_tree/model.txt"
    result = scan_model_file(model_file)
    assert len(result["trees"]) == 1
    tree_0 = result["trees"][0]
    assert tree_0["num_leaves"] == 4
