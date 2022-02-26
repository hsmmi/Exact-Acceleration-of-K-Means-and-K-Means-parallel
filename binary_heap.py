import heapq
from numpy import ndarray


class BinaryHeap():
    def __init__(self, value: ndarray = None) -> None:
        """ Build binary heap tree

        Creates a binary heap tree empty or with your input values

        Args:
            value: nx1 array
        """
        self.heap = list(zip(value, range(value.shape[0])))
        heapq.heapify(self.heap)

    def push(self, value: int, index: int):
        """ Build binary heap tree

        Args:
            index (int): index of value
            value (float): value that want to add to tree
        """
        heapq.heappush(self.heap, (value, index))

    def pop(self):
        """ Pop the minimum value

        Pop the node with minimum value(root) from tree and remove the node

        Returns:
            value (float): The minimum value in tree
        """
        return heapq.heappop(self.heap)[1]

    def peek(self):
        """ Peek the minimum value

        Peek the node with minimum value(root) from tree but node
        still is in tree

        Returns:
            value (float): The minimum value in tree
        """
        return self.heap[0][1]
