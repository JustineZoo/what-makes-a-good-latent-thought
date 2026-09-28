"""Synthetic reasoning task registry (torch-free)."""
from __future__ import annotations

from typing import Dict, Type

from .base import Batch, Task
from .vocab import Vocab
from .function_compose import FunctionCompose
from .graph_reach import GraphReach

TASKS: Dict[str, Type[Task]] = {
    "function_compose": FunctionCompose,
    "graph_reach": GraphReach,
}


def make_task(name: str, **kwargs) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name](**kwargs)


__all__ = ["Batch", "Task", "Vocab", "make_task", "TASKS"]
