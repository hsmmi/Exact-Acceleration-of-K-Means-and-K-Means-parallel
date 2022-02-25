import numpy as np
import vptree


def euclidean(p1: tuple, p2: tuple) -> float:
    return np.linalg.norm(p1[0] - p2[0])


class nns:
    def __init__(self, points: np.ndarray or vptree.VPTree) -> None:
        if type(points) == np.ndarray:
            points = list(zip(points, range(points.shape[0])))
            self.tree = vptree.VPTree(points, euclidean)
        else:
            self.tree = points

    # line 1 algorithm 4
    def get_child(self, low: bool):
        # line 2 - 4 algorithm 4
        if low is True:
            return nns(self.tree.left)
        return nns(self.tree.right)

    # line 5 algorithm 4
    def search(self, r: float, tau: float, low: bool):
        # line 6 - 9 algorithm 4
        if low is True:
            a, b = self.tree.left_min, self.tree.left_max
        else:
            a, b = self.tree.right_min, self.tree.right_max

        # line 10 algorithm 4
        if(a - tau < r and r < b + tau):
            return True
        return False

    # line 11 algorithm 4
    def best(self, tau: float, tau_p: float, id: tuple, id_p: tuple):
        # line 12 - 14 algorithm 4
        if tau < tau_p:
            return tau, id
        return tau_p, id_p

    # line 15 algorithm 4
    def nearest(self, q: tuple, tau: float):
        # line 16 algorithm 4
        p = self.tree.vp
        r = euclidean(p, q)
        # line 17 algorithm 4
        tau, id = self.best(tau, r, q, p)
        # line 18 algorithm 4
        m = (self.tree.left_max + self.tree.right_min) / 2
        # line 19 algorithm 4
        low = r < m
        # line 20 - 22 algorithm 4
        if self.search(r, tau, low):
            tau_p, id_p = self.get_child(low).nearest(q, tau)
            tau, id = self.best(tau, tau_p, id, id_p)
        # line 23 - 25 algorithm 4
        if self.search(r, tau, not low):
            tau_p, id_p = self.get_child(not low).nearest(q, tau)
            tau, id = self.best(tau, tau_p, id, id_p)
        # line 26 algorithm 4
        return tau, id

    # line 27 algorithm 4
    def nearest_in_range(self, q, max_range):
        # line 28 algorithm 4
        return self.nearest((q, -1), max_range)
