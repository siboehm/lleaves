def parse_model_file(file_path):
    res = {"trees": []}

    with open(file_path, "r") as f:
        # List of blocks we expect:
        # 1* General Information
        # N* Tree, one block for each tree
        # 1* 'end of trees'
        # 1* Feature importances
        # 1* Parameters
        # 1* 'end of parameters'
        # 1* 'pandas_categorical:XXXXX'
        lines = _get_next_block_of_lines(f)
        assert lines[0] == "tree" and lines[1].startswith(
            "version="
        ), f"{file_path} is not a LightGBM model file"
        res["general_info"] = _struct_from_block(lines, INPUT_PARSED_KEYS)

        lines = _get_next_block_of_lines(f)
        while lines:
            if lines[0].startswith("Tree="):
                res["trees"].append(_struct_from_block(lines, TREE_PARSED_KEYS))
            else:
                assert lines[0] == "end of trees"
                return res
            lines = _get_next_block_of_lines(f)
    raise ValueError(f"Ill formatted file {file_path}")


def _get_next_block_of_lines(file):
    # the only function where we advance file_offset
    result = []
    line = file.readline()
    while line == "\n":
        line = file.readline()
    while line != "\n" and line != "":
        result.append(line.strip())
        line = file.readline()
    return result


INPUT_PARSED_KEYS = {"max_feature_idx": int, "version": str}
TREE_PARSED_KEYS = {
    "Tree": int,
    "num_leaves": int,
    "split_feature": list[int],
    "threshold": list[float],
    "decision_type": list[int],
    "left_child": list[int],
    "right_child": list[int],
    "leaf_value": list[float],
}


def _struct_from_block(lines: list[str], parsed_keys: dict):
    """
    Parses a block (= list of lines) into a key: value struct
    @param lines: list of lines in the block
    @param parsed_keys: dict with 'key': 'type of value' of keys to parse
    """
    struct = {}
    for line in lines:
        # initial line in file
        if line == "tree":
            continue

        key, value = line.split("=")
        if key in parsed_keys.keys():
            typ = parsed_keys[key]
            if typ == list[int]:
                parsed_value = [int(x) for x in value.split(" ")]
            elif typ == list[float]:
                parsed_value = [float(x) for x in value.split(" ")]
            elif typ == int:
                parsed_value = int(value)
            else:
                assert typ == str
                parsed_value = value
            struct[key] = parsed_value
    # make sure we managed to parse all desired keys
    assert struct.keys() == parsed_keys.keys()
    return struct
