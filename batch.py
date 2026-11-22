"""Batch processing: apply the current edit chain to many images using a FIFO queue."""
from chain import chain_matrix
from fifo_queue import FifoQueue
from warp import inverse_warp


def build_queue(named_images):
    q = FifoQueue()
    for name, arr in named_images:
        q.enqueue((name, arr))
    return q


def process_queue(jobs, chain, interpolation="bilinear", expand=True):
    """Dequeue images one by one (first in, first out) and warp each.

    The chain is rebuilt for each image's own size, so every edit happens
    about that image's own centre."""
    results = []
    while not jobs.is_empty():
        name, arr = jobs.dequeue()
        h, w = arr.shape[:2]
        M = chain_matrix(chain, w, h)
        results.append((name, inverse_warp(arr, M, interpolation, expand)))
    return results
