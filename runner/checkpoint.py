import io
import logging
import os
import os.path as osp
import pkgutil
import re
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
from torch.optim import Optimizer

from ..fileio import FileClient
from ..fileio import load as load_file
from ..parallel import is_module_wrapper
from ..utils import digit_version, load_url, mkdir_or_exist
from .dist_utils import get_dist_info


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: dict = {}

    @classmethod
    def _register_scheme(cls,
                         prefixes: Union[str, List, Tuple],
                         loader: Callable,
                         force: bool = False) -> None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))
