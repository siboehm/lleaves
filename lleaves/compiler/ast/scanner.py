import json
import os

"""
The Scanner is responsible for iterating over the model.txt and transforming it into a more
usable representation.
It doesn't implement any transformations (expect for type casting).
"""


def scan_for_pandas_categorical(file_path):
    """
    Scan the model.txt from the back to extract the 'pandas_categorical' field.

    This is a list of lists that stores the ordering of categories from the pd.DataFrame used for training.
    Storing this list is necessary as LightGBM encodes categories as integer indices and we need to guarantee that
    the mapping (<category string> -> <integer idx>) is the same during inference as it was during training.

    Example (pandas categoricals were present in training):
    pandas_categorical:[["a", "b", "c"], ["b", "c", "d"], ["w", "x", "y", "z"]]
    Example (no pandas categoricals during training):
    pandas_categorical:[] OR pandas_categorical=null

    LightGBM generates this list of lists like so:
      pandas_categorical = [list(df[col].cat.categories) for col in df.select_dtypes(include=['category']).columns]
    and stores it via json.dump

    :param file_path: path to model.txt
    :return: list of list. For each pd.categorical column encountered during training, a list of the categories.
    """
    pandas_key = "pandas_categorical:"
    max_offset = os.path.getsize(file_path)
    stepsize = min(1024, max_offset - 1)
    current_offset = stepsize
    lines = []
    # seek backwards from end of file until we have two lines
    # the (pen)ultimate line should be pandas_categorical:XXX
    with open(file_path, "rb") as f:
        while len(lines) < 2 and current_offset < max_offset:
            if current_offset > max_offset:
                current_offset = max_offset
            # read <current_offset>-many Bytes from end of file
            f.seek(-current_offset, os.SEEK_END)
            lines = f.readlines()
            current_offset *= 2

    # pandas_categorical has to be present in the ultimate or penultimate line. Else the model.txt is malformed.
    if len(lines) >= 2:
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
        res["general_info"] = _scan_block(lines, INPUT_SCAN_KEYS)
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
    res["pandas_categorical"] = scan_for_pandas_categorical(file_path)
    return res


def _scan_tree(lines):
    struct = _scan_block(lines, TREE_SCAN_KEYS)
    return struct


def _get_next_block_of_lines(file):
    # the only function where the position inside the file is advanced
    result = []
    line = file.readline()
    while line == "\n":
        line = file.readline()
    while line != "\n" and line != "":
        result.append(line.strip())
        line = file.readline()
    return result


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

        scanned_key, scanned_value = line.split("=")
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
