import numpy as np
from functools import wraps
from time import time
import pickledb
from scipy.spatial.distance import cdist


def tlogger(f):
    def wrapped(*args, **kwargs):
        if type(args[0]) == tuple:
            wrapped.sum += 1
        else:
            if len(args[0].shape) == 1:
                args = (args[0].reshape((1, -1)), args[1])
            if len(args[1].shape) == 1:
                args = (args[0], args[1].reshape((1, -1)))
            x = args[0].shape[0]
            y = args[1].shape[0]
            wrapped.sum += x * y
        return f(*args, **kwargs)

    wrapped.sum = 0
    return wrapped


def logger(f):
    #  computations logger
    @wraps(f)
    def wrap(*args, **kw):
        x = args[0].shape[0]
        y = args[1].shape[0]
        num_of_computations = x * y
        db = pickledb.load("log.db", False)
        db.set("c_sum", db.get("c_sum") + num_of_computations)
        c = db.get("computations")
        db.set("computations", [*c, num_of_computations])
        db.dump()
        return f(*args, **kw)

    return wrap


def db_init():
    db = pickledb.load("log.db", False)
    db.set("computations", [])
    db.set("c_sum", 0)
    db.dump()
    return True


def get_log():
    try:
        db = pickledb.load("log.db", False)
        return db.get("computations"), db.get("c_sum")
    except Exception as e:
        print(e)


@tlogger
def distance(X: np.ndarray, Y: np.ndarray):
    """Using scipy to find distance of each pair (x, y)
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
    return cdist(X, Y)


def distance2(X: np.ndarray, Y: np.ndarray):
    """Using numpy to find distance of each pair (x, y)
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
        print(f"func:{f.__qualname__} took: {te-ts} sec")

        return result

    return wrap
