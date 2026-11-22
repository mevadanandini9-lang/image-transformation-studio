"""Queue (FIFO) built on a singly linked list. Used for BATCH processing."""


class _Node:
    __slots__ = ("value", "next")

    def __init__(self, value):
        self.value, self.next = value, None


class FifoQueue:
    def __init__(self):
        self._head = self._tail = None
        self._size = 0

    def enqueue(self, value):       # O(1)
        node = _Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def dequeue(self):              # O(1)
        if self._head is None:
            raise IndexError("dequeue from empty queue")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return node.value

    def is_empty(self):
        return self._size == 0

    def __len__(self):
        return self._size
