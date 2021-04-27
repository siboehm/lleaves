def decision_idx_to_llvmlite_str(idx):
    if idx == 1:
        return "=="
    elif idx == 2:
        return "<="
    else:
        raise ValueError(f"decision type {idx} not yet supported")
