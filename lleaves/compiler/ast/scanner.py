import json
import os


def scan_pandas_categorical(file_path):
    pandas_key = "pandas_categorical:"
    offset = -len(pandas_key)
    max_offset = -os.path.getsize(file_path)
    # seek backwards from end of file until we have to lines
    # the (pen)ultimate line should be pandas_categorical:XXX
    with open(file_path, "rb") as f:
        while True:
            if offset < max_offset:
                offset = max_offset
            f.seek(offset, os.SEEK_END)
            lines = f.readlines()
            if len(lines) >= 2:
                break
            offset *= 2
    last_line = lines[-1].decode().strip()
    if not last_line.startswith(pandas_key):
        last_line = lines[-2].decode().strip()
    if last_line.startswith(pandas_key):
        return json.loads(last_line[len(pandas_key) :])
    raise ValueError("Ill formatted model file!")


def scan_model_file(file_path, general_info_only=False):
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
        res["general_info"] = _struct_from_block(lines, INPUT_SCAN_KEYS)
        if general_info_only:
            return res

        lines = _get_next_block_of_lines(f)
        while lines:
            if lines[0].startswith("Tree="):
                res["trees"].append(_scan_tree(lines))
            else:
                assert lines[0] == "end of trees"
                break
            lines = _get_next_block_of_lines(f)
    res["pandas_categorical"] = scan_pandas_categorical(file_path)
    return res


def _scan_tree(lines):
    struct = _struct_from_block(lines, TREE_SCAN_KEYS)
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


class ScannedValue:
    def __init__(self, type: type, is_list=False, null_ok=False):
        self.type = type
        self.is_list = is_list
        self.null_ok = null_ok


INPUT_SCAN_KEYS = {
    "max_feature_idx": ScannedValue(int),
    "version": ScannedValue(str),
    "feature_infos": ScannedValue(str, True),
    "objective": ScannedValue(str, True),
}
TREE_SCAN_KEYS = {
    "Tree": ScannedValue(int),
    "num_leaves": ScannedValue(int),
    "num_cat": ScannedValue(int),
    "split_feature": ScannedValue(int, True),
    "threshold": ScannedValue(float, True),
    "decision_type": ScannedValue(int, True),
    "left_child": ScannedValue(int, True),
    "right_child": ScannedValue(int, True),
    "leaf_value": ScannedValue(float, True),
    "cat_threshold": ScannedValue(int, True, True),
    "cat_boundaries": ScannedValue(int, True, True),
}


def _struct_from_block(lines: list, keys_to_scan: dict):
    """
    Scans a block (= list of lines), produces a key: value struct
    @param lines: list of lines in the block
    @param keys_to_scan: dict with 'key': 'type of value' of keys to scan for
    """
    struct = {}
    for line in lines:
        # initial line in file
        if line == "tree":
            continue

        key, value = line.split("=")
        if key in keys_to_scan.keys():
            value_type = keys_to_scan[key]
            if value_type.is_list:
                if value:
                    parsed_value = [value_type.type(x) for x in value.split(" ")]
                else:
                    parsed_value = []
            else:
                parsed_value = value_type.type(value)
            struct[key] = parsed_value

    missing_keys = keys_to_scan.keys() - struct.keys()
    for key in missing_keys:
        value = keys_to_scan[key]
        assert value.null_ok, f"Non-nullable key {key} wasn't found"
        struct[key] = None
    return struct
