CATEGORICAL_EQ_DECISION = 1
NUMERCIAL_DECISION = 2


def decision_idx_to_llvmlite_str(idx):
    if idx == 1:
        return "=="
    elif idx == 2:
        return "<="
    else:
        raise ValueError(f"decision type {idx} not yet supported")


def calc_pymode_cat_thresholds(threshold):
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
