import heapq
from numpy import ndarray


class Binary_heap():
    def __init__(self, value: ndarray) -> None:
        """
        Parameter:
            value: nx1 array
        """
        self.heap = list(zip(value, range(value.shape[0])))
        heapq.heapify(self.heap)

    def push(self, value: int, index: int):
        heapq.heappush(self.heap, (value, index))

    def pop(self):
        return heapq.heappop(self.heap)[1]

    def peek(self):
        return self.heap[0][1]
