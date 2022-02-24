import numpy as np


def distance(x: np.ndarray, Y: np.ndarray):
    """
    Parameter:
        x: nxd data(s)
        Y: mxd point(s)

    return
        d: nxm ndarray contain distance of data to each point
    """
    assert len(x.shape) == 2, 'x should be 1xd ndarray'
    assert len(x.shape) == 2 and Y.shape[1] == x.shape[1], \
        'y should be in x space'
    return np.linalg.norm((x[:, np.newaxis, :]-Y), axis=-1)


def new_seed(X, L, probability):
    """
    Parameters:
       X: nxd
       probability: nx1
       L: number of new seed

    Returns:
        z: ndarray Lxd contain L new seed(s)
    """
    assert len(X.shape) == 2, 'X should be 2-D array'
    assert len(probability.shape) == 2 and probability.shape[1] == 1,\
        'probability should be 2-D array nx1'
    assert probability.shape[0] == X.shape[0],\
        'first dimantion on X, L should be the same'
    return X[np.random.choice(
        range(len(probability)), L, p=probability.flatten()), :]
