import numpy as np
import vptree


def euclidean(p1: tuple, p2: tuple) -> float:
    """Distance of two points

    Args:
        p1 (ndarray, int): point in our tree
        p2 (ndarray, int): point in our tree

    Returns:
        float: distance of two points from eachother
    """
    return np.linalg.norm(p1[0] - p2[0])


class Point:
    def __init__(self, loc: np.ndarray, index: int) -> None:
        self.loc = loc
        self.index = index


class NNS:
    def __init__(
        self, points: np.ndarray or vptree.VPTree, id: np.ndarray = None
    ) -> None:
        """To find nearest neighbor in O(log n)

        Build Vantage Point-tree (VP-tree) from you points

        Points can be ndarray, it'll build VPTree with each point
        like (point, id)
        or be VPTree that module will use it as tree

        Args:
            points(ndarray|VPTree)
        """
        if type(points) == np.ndarray:
            if id is None:
                points = list(zip(points, range(points.shape[0])))
            else:
                points = list(zip(points, id))
            self.tree = vptree.VPTree(points, euclidean)
        else:
            self.tree = points

    # Line 1 algorithm 4
    def get_child(self, low: bool):
        """Return a child of root

        If low be True then return the left child otherwise
        return the right child

        Args:
            low (bool): can be True or False

        Returns:
            child (NNS): on of the root child depends on arg(low)
        """
        # Line 2 - 4 algorithm 4
        # Return object NNS with root child
        if low is True:
            return NNS(self.tree.left)
        return NNS(self.tree.right)

    # Line 5 algorithm 4
    def search(self, r: float, tau: float, low: bool):
        """Should I search this child?
        Is it possible nearest neibour be in this child due to
        r(distance of query from root) and tau"""
        # Line 6 - 9 algorithm 4
        # Find the range of points in the child from root
        if low is True:
            a, b = self.tree.left_min, self.tree.left_max
        else:
            a, b = self.tree.right_min, self.tree.right_max

        # Line 10 algorithm 4
        # Is our query(with distance r from root) in this child?
        if a - tau < r and r < b + tau:
            return True
        return False

    # Line 11 algorithm 4
    def best(self, tau: float, tau_p: float, id: tuple, id_p: tuple):
        """Which one is closer?

        If low be True then return the left child otherwise
        return the right child

        Args:
            tau (float): distance id from root
            tau_p (float): distance id_p from root
            id (tuple): is a point
            id_p (tuple): is a point

        Returns:
            tau (float), id (tuple)
        """
        # Line 12 - 14 algorithm 4
        # Reruen closer point
        if tau < tau_p:
            return tau, id
        return tau_p, id_p

    # Line 15 algorithm 4
    def nearest(self, q: tuple, tau: float) -> tuple:
        # Line 16 algorithm 4
        # Root of tree
        p = self.tree.vp
        # Find distace of query from root
        r = euclidean(p, q)
        # Line 17 algorithm 4
        # Update distace and id of nearest point
        tau, id = self.best(tau, r, q, p)
        # Line 18 algorithm 4
        # Find meddian distace that seprate point to two half
        m = (self.tree.left_max + self.tree.right_min) / 2
        # Line 19 algorithm 4
        # Search near or far first?
        low = r < m
        # Line 20 - 22 algorithm 4
        if self.search(r, tau, low):
            # Find nearest in child
            tau_p, id_p = self.get_child(low).nearest(q, tau)
            # Update distace and id of nearest point
            tau, id = self.best(tau, tau_p, id, id_p)

        # Line 23 - 25 algorithm 4
        if self.search(r, tau, not low):
            # Find nearest in child
            tau_p, id_p = self.get_child(not low).nearest(q, tau)
            # Update distace and id of nearest point
            tau, id = self.best(tau, tau_p, id, id_p)
        # Line 26 algorithm 4
        return tau, id

    # Line 27 algorithm 4
    def nearest_in_range(self, q: np.ndarray, max_range: float) -> tuple:
        """Get point q and find closest point in range max range

        Args:
            q (ndarray): one sample in space
            max_range (float): in range max_range from q with
                euclidean distance

        Returns:
            ret (distance, id): ret[0] is distance nearest point form q
            ret[1] is id of nearest point
        """
        # Line 28 algorithm 4
        # Call nearest function to search in distance max range with id -1
        # If there is no point in our range return -1
        tau, id = self.nearest((q, -1), max_range)
        return tau, id[1]
