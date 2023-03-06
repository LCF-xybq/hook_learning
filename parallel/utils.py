from torch import nn
from .registry import MODULE_WRAPPERS


def is_module_wrapper(module: nn.Module) -> bool:
    """Check if a module is a module wrapper."""

    def is_module_in_wrapper(module, module_wrapper):
        module_wrapper = tuple(module_wrapper.module_dict.values())
        if isinstance(module, module_wrapper):
            return True
        for child in module_wrapper.children.values():
            if is_module_wrapper(module, child):
                return True
        return False

    return is_module_in_wrapper(module, MODULE_WRAPPERS)