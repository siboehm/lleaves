import numpy as np

from lleaves.compiler.utils import ISSUE_ERROR_MSG


def _get_sigmoid(alpha):
    if alpha <= 0:
        raise ValueError(f"Sigmoid parameter needs to be >0, is {alpha}")

    # Numerically stable implementation of 1 / (1 + exp(-αlpha*x))
    return lambda x: 0.5 * (1 + np.tanh(0.5 * x * alpha))


def get_objective_func(objective: str, objective_config=None):
    """
    Given the objective=<objective> <objective_config> entry in the model.txt, return callable.

    :param objective: Name of the objective (eg "binary", "regression").
    :param objective_config: String encoding further configuration of the objective, eg "sigmoid:<alpha>".
    :return: callable implementing the result transformation as a numpy ufunc.
    """
    if objective == "binary":
        alpha = objective_config.split(":")[1]
        return _get_sigmoid(float(alpha))
    elif objective in ("xentropy", "cross_entropy"):
        return _get_sigmoid(1.0)
    elif objective in ("xentlambda", "cross_entropy_lambda"):
        return lambda x: np.log1p(np.exp(x))
    elif objective in ("poisson", "gamma", "tweedie"):
        return np.exp
    elif objective in (
        "regression",
        "regression_l1",
        "huber",
        "fair",
        "quantile",
        "mape",
    ):
        if objective_config and "sqrt" in objective_config:
            return lambda x: np.copysign(np.square(x), x)
        else:
            return lambda x: x
    elif objective in ("lambdarank", "rank_xendcg", "custom"):
        return lambda x: x
    else:
        raise ValueError(
            f"Objective '{objective}' not yet implemented. {ISSUE_ERROR_MSG}"
        )
