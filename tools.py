import numpy as np
from functools import wraps
from time import time


def distance(X: np.ndarray, Y: np.ndarray):
    """
    Parameters:
        X: nxd data(s)
        Y: mxd point(s)

    Returns:
        d: nxm ndarray contain distance of data to each point
    """
    if len(X.shape) == 1:
        X = X.reshape((1, -1))
    if len(Y.shape) == 1:
        Y = Y.reshape((1, -1))
    assert len(X.shape) == 2, "X should be nxd ndarray"
    assert (
        len(Y.shape) == 2 and Y.shape[1] == X.shape[1]
    ), "y should be in X space"
    return np.linalg.norm((X[:, np.newaxis, :] - Y), axis=-1)


def new_seed(X, L, probability):
    """
    Parameters:
       X: nxd
       probability: nx1
       L: number of new seed

    Returns:
        z: ndarray Lxd contain L new seed(s)
    """
    assert len(X.shape) == 2, "X should be 2-D array"
    assert (
        len(probability.shape) == 2 and probability.shape[1] == 1
    ), "probability should be 2-D array nx1"
    assert (
        probability.shape[0] == X.shape[0]
    ), "first dimantion on X, probability should be the same"
    probability = probability.flatten()
    return X[np.random.choice(range(len(probability)), L, p=probability), :]


def execution_time(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {te-ts} sec")

        return result

    return wrap
