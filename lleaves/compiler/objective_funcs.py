import numpy as np

from lleaves.compiler.utils import ISSUE_ERROR_MSG


def _get_sigmoid(alpha):
    if alpha <= 0:
        raise ValueError(f"Sigmoid parameter needs to be >0, is {alpha}")

    # Numerically stable implementation of 1 / (1 + exp(-αlpha*x))
    return lambda x: 0.5 * (1 + np.tanh(0.5 * x * alpha))


def get_objective_func(obj):
    """
    Given the objective=obj entry in the model.txt, return callable
    :return: callable implementing the correct result transformation as np.ufunc
    """
    if obj[0] == "binary":
        func, alpha = obj[1].split(":")
        f = _get_sigmoid(float(alpha))
    elif obj[0] in ("xentropy", "cross_entropy"):
        f = _get_sigmoid(1.0)
    elif obj[0] in ("xentlambda", "cross_entropy_lambda"):

        def f(x):
            return np.log1p(np.exp(x))

    elif obj[0] in ("poisson", "gamma", "tweedie"):
        f = np.exp
    elif obj[0] in ("regression", "regression_l1", "huber", "fair", "quantile", "mape"):
        if "sqrt" in obj[1:]:

            def f(x):
                return np.copysign(np.square(x), x)

        else:

            def f(x):
                return x

    elif obj[0] in ("lambdarank", "rank_xendcg", "custom"):

        def f(x):
            return x

    else:
        raise ValueError(f"Objective '{obj[0]}' not yet implemented. {ISSUE_ERROR_MSG}")
    return f
