import copy
from typing import Optional

from registry_learning import Registry


RUNNERS = Registry('runner')
RUNNERS_BUILDERS = Registry('runner builder')

def build_runner_constructor(cfg: dict):
    return RUNNERS_BUILDERS.build(cfg)


def build_runner(cfg: dict, default_args: Optional[dict] = None):
    runner_cfg = copy.deepcopy(cfg)
    constructor_type = runner_cfg.pop('constructor',
                                      'DefaultRunnerConstructor')
    runner_constructor = build_runner_constructor(
        dict(
            type=constructor_type,
            runner_cfg=runner_cfg,
            default_args=default_args))
    runner = runner_constructor()
    return runner
