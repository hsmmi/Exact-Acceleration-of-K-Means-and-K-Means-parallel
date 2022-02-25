import numpy as np


class Node:
    def __init__(self) -> None:
        self.root = None
        self.p = None
        self.left_child = None  # nodes with distance more than the threshold
        self.near_low = None  # closest distance in left child
        self.near_high = None  # farthest distance in left child
        self.right_child = None  # nodes with distance less than the threshold
        self.far_low = None  # closest distance in right child
        self.far_high = None  # farthest distance in right child


class VPTree:
    def __init__(self) -> None:
        self.tau = None
        self.max_leaf_size = 5
        self.vp_index = None  # vp node
