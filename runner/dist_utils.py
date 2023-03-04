import functools
import os
import socket
import subprocess
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def allreduce_params(params: List[torch.nn.Parameter],
                     coalesce: bool = True,
                     bucket_size_mb: int = -1) -> None:
    """Allreduce parameters.

    Args:
        params (list[torch.nn.Parameter]): List of parameters or buffers
            of a model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    pass
