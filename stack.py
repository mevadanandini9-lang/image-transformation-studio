"""Stack (LIFO) built on a singly linked list. Used for UNDO (and redo)."""


class _Node:
    __slots__ = ("value", "below")

    def __init__(self, value, below):
        self.value, self.below = value, below


class Stack:
    def __init__(self):
        self._top = None
        self._size = 0

    def push(self, value):          # O(1)
        self._top = _Node(value, self._top)
        self._size += 1

    def pop(self):                  # O(1)
        if self._top is None:
            raise IndexError("pop from empty stack")
        node = self._top
        self._top = node.below
        self._size -= 1
        return node.value

    def peek(self):                 # O(1)
        if self._top is None:
            raise IndexError("peek at empty stack")
        return self._top.value

    def is_empty(self):
        return self._size == 0

    def clear(self):
        self._top, self._size = None, 0

    def items(self):
        """Values from bottom to top (oldest first). O(n)."""
        out, node = [], self._top
        while node:
            out.append(node.value)
            node = node.below
        return out[::-1]

    def __len__(self):
        return self._size
