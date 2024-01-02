"""
The Scanner is responsible for iterating over the model.txt and transforming it into a more
usable representation.
It doesn't implement any transformations (expect for type casting).
"""


from io import StringIO


def scan_model_file(model_str, general_info_only=False):
    res = {"trees": []}

    def read_blocks(model_str):
        stream = StringIO(model_str)
        while True:
            lines = _get_next_block_of_lines(stream)
            if lines:
                yield lines
            else:
                break

    blocks = read_blocks(model_str)
    # List of blocks we expect:
    # 1* General Information
    # N* Tree, one block for each tree
    # 1* 'end of trees'
    # followed by these ignored blocks:
    # 1* Feature importances
    # 1* Parameters
    # 1* 'end of parameters'
    # 1* 'pandas_categorical:XXXXX'

    general_info_block = next(blocks)
    assert general_info_block[0] == "tree" and general_info_block[1].startswith(
        "version="
    ), f"supplied model is not a valid LightGBM model definition"
    res["general_info"] = _scan_block(general_info_block, INPUT_SCAN_KEYS)
    if general_info_only:
        return res

    for block in blocks:
        if block[0].startswith("Tree="):
            res["trees"].append(_scan_tree(block))
        else:
            assert block[0] == "end of trees"
            break
    return res


def _scan_tree(lines):
    struct = _scan_block(lines, TREE_SCAN_KEYS)
    return struct


def _get_next_block_of_lines(stream):
    # the only function where the position inside the file is advanced
    result = []
    line = stream.readline()
    while line == "\n":
        line = stream.readline()
    while line != "\n" and line != "":
        result.append(line.strip())
        line = stream.readline()
    return result


class ScannedValue:
    def __init__(self, type: type, is_list=False, null_ok=False):
        self.type = type
        self.is_list = is_list
        self.null_ok = null_ok


INPUT_SCAN_KEYS = {
    "max_feature_idx": ScannedValue(int),
    "num_class": ScannedValue(int),
    "num_tree_per_iteration": ScannedValue(int),
    "version": ScannedValue(str),
    "feature_infos": ScannedValue(str, True),
    "objective": ScannedValue(str, True),
    "average_output": ScannedValue(bool, null_ok=True),
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


def _scan_block(lines: list, items_to_scan: dict):
    """
    Scans a block (= list of lines) into a key: value map.
    :param lines: list of lines in the block
    :param items_to_scan: dict with 'key': 'type of value' of keys to scan for
    :return: dict with a key-value pair for each key in items_to_scan. Raises RuntimeError
        if a non-nullable value from items_to_scan wasn't found in the block.
    """
    result_map = {}
    for line in lines:
        # initial line in file
        if line == "tree":
            continue

        line_split = line.split("=")
        if len(line_split) == 2:
            scanned_key, scanned_value = line.split("=")
        else:
            assert len(line_split) == 1, f"Unexpected line {line}"
            scanned_key, scanned_value = line_split[0], True

        target_type = items_to_scan.get(scanned_key)
        if target_type is None:
            continue
        if target_type.is_list:
            if scanned_value:
                parsed_value = [target_type.type(x) for x in scanned_value.split(" ")]
            else:
                parsed_value = []
        else:
            parsed_value = target_type.type(scanned_value)
        result_map[scanned_key] = parsed_value

    expected_keys = {k for k, v in items_to_scan.items() if not v.null_ok}
    missing_keys = expected_keys - result_map.keys()
    if missing_keys:
        raise RuntimeError(f"Missing non-nullable keys {missing_keys}")
    return result_map
