"""Run one function over jobs in order, in this process or a pool of worker processes.

Used by phase.py to run steps a_open_sources to e_write once per data source. Results come back in job order
whatever the number of workers, and each source has its own seeds, so outputs never depend on the worker count.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import TypeVar

J = TypeVar("J")
R = TypeVar("R")


def run_in_order(fn: Callable[[J], R], jobs: Sequence[J], workers: int) -> list[R]:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as pool:
        return list(pool.map(fn, jobs))
