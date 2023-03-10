from registry_learning import Registry
from torch.nn.parallel import DataParallel, DistributedDataParallel


MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)
