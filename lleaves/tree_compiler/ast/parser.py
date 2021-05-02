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
                res["trees"].append(_parse_tree(lines))
            else:
                assert lines[0] == "end of trees"
                return res
            lines = _get_next_block_of_lines(f)
    raise ValueError(f"Ill formatted file {file_path}")


def _parse_tree(lines):
    struct = _struct_from_block(lines, TREE_PARSED_KEYS)
    return struct


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


def cat_args_bitmap(arr):
    # Feature infos for floats look like [x.xxxx:y.yyyy]
    # for categoricals like X:Y:Z:
    return [not val.startswith("[") for val in arr]


class ParsedValue:
    def __init__(self, type: type, is_list=False, null_ok=False):
        self.type = type
        self.is_list = is_list
        self.null_ok = null_ok


INPUT_PARSED_KEYS = {
    "max_feature_idx": ParsedValue(int),
    "version": ParsedValue(str),
    "feature_infos": ParsedValue(str, True),
}
TREE_PARSED_KEYS = {
    "Tree": ParsedValue(int),
    "num_leaves": ParsedValue(int),
    "num_cat": ParsedValue(int),
    "split_feature": ParsedValue(int, True),
    "threshold": ParsedValue(float, True),
    "decision_type": ParsedValue(int, True),
    "left_child": ParsedValue(int, True),
    "right_child": ParsedValue(int, True),
    "leaf_value": ParsedValue(float, True),
    "cat_threshold": ParsedValue(int, True, True),
    "cat_boundaries": ParsedValue(int, True, True),
}


def _struct_from_block(lines: list, keys_to_parse: dict):
    """
    Parses a block (= list of lines) into a key: value struct
    @param lines: list of lines in the block
    @param keys_to_parse: dict with 'key': 'type of value' of keys to parse
    """
    struct = {}
    for line in lines:
        # initial line in file
        if line == "tree":
            continue

        key, value = line.split("=")
        if key in keys_to_parse.keys():
            value_type = keys_to_parse[key]
            if value_type.is_list:
                parsed_value = [value_type.type(x) for x in value.split(" ")]
            else:
                parsed_value = value_type.type(value)
            struct[key] = parsed_value

    missing_keys = keys_to_parse.keys() - struct.keys()
    for key in missing_keys:
        value = keys_to_parse[key]
        assert value.null_ok, f"Non-nullable key {key} wasn't found"
        struct[key] = None
    return struct
