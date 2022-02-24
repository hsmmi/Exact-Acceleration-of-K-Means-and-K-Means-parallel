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


def new_seed(X, L, dist):
    """
    Parameters:
       X: nxd
       dist: nx1
       L: number of new seed

    Returns:
        z: ndarray Lxd contain L new seed(s)
    """
    return X[np.random.choice(range(len(dist)), L, p=dist), :]
