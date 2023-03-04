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
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            """
            Reduces the tensor data across all machines 
            in such a way that all get the final result.
            """
            dist.all_reduce(tensor.div_(world_size))


def _allreduce_coalesced(tensors: torch.Tensor,
                         world_size: int,
                         bucket_size_mb: int = -1) -> None:
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        """
        Group tensors into chunks. 
        This generator yields a chunk at each time,
        each containing tensors of same type up 
        to certain byte limit in total size.
    """
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        """
        Flatten dense tensors into a contiguous 1D buffer. 
        Assume tensors are of same dense type.
        """
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)