import logging

ISSUE_ERROR_MSG = "Please file an issue at https://github.com/siboehm/lleaves."


class DecisionType:
    CAT_MASK = 1
    DEFAULT_LEFT_MASK = 2

    def __init__(self, idx):
        if idx not in [1, 2, 9]:
            logging.warning(
                f"Decision type {idx} not yet tested for. {ISSUE_ERROR_MSG}"
            )
        self.idx = idx

    @property
    def is_categorical(self):
        return bool(self.idx & DecisionType.CAT_MASK)

    @property
    def is_default_left(self):
        assert not self.is_categorical
        return bool(self.idx & DecisionType.DEFAULT_LEFT_MASK)

    def __str__(self):
        if self.is_categorical:
            return "=="
        else:
            return "<=" if self.is_default_left else ">="


def bitset_to_py_list(threshold):
    """
    :param threshold: An integer representing a bitvector
    :return: list of ints, each item corresponding to entry in the bitvector
    """
    thresh = threshold
    cat_thresholds = []
    i = 0
    while thresh > 0:
        tmp = thresh % 2
        thresh = thresh // 2
        if tmp:
            cat_thresholds.append(i)
        i += 1
    return cat_thresholds
