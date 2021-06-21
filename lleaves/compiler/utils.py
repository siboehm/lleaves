from enum import Enum

ISSUE_ERROR_MSG = "Please file an issue at https://github.com/siboehm/lleaves."


class MissingType(Enum):
    MNone = 0
    MZero = 1
    MNaN = 2


class DecisionType:
    CAT_MASK = 1
    DEFAULT_LEFT_MASK = 2

    def __init__(self, idx):
        if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            raise ValueError(
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

    @property
    def missing_type(self):
        missing_type = (self.idx >> 2) & 3
        return MissingType(missing_type)

    def __str__(self):
        if self.is_categorical:
            return "=="
        else:
            return "<="
