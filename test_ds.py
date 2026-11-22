import pytest

from ds.fifo_queue import FifoQueue
from ds.stack import Stack


def test_stack_lifo():
    s = Stack()
    for x in (1, 2, 3):
        s.push(x)
    assert (s.pop(), s.pop(), s.peek(), len(s)) == (3, 2, 1, 1)

def test_stack_empty_raises_and_clear():
    s = Stack()
    with pytest.raises(IndexError):
        s.pop()
    s.push(1); s.clear()
    assert s.is_empty()

def test_stack_items_bottom_to_top():
    s = Stack()
    for x in "abc":
        s.push(x)
    assert s.items() == ["a", "b", "c"]

def test_queue_fifo():
    q = FifoQueue()
    for x in (1, 2, 3):
        q.enqueue(x)
    assert (q.dequeue(), q.dequeue(), len(q)) == (1, 2, 1)

def test_queue_empty_and_reuse_after_emptying():
    q = FifoQueue()
    with pytest.raises(IndexError):
        q.dequeue()
    q.enqueue(1); q.dequeue(); q.enqueue(2)
    assert q.dequeue() == 2 and q.is_empty()
