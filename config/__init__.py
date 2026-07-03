# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

try:
    from .defaults import _C as cfg
    from .defaults import _C as cfg_test
    from .defaults_base import _C as cfg_base
except ModuleNotFoundError as exc:
    if exc.name != "yacs":
        raise
    cfg = None
    cfg_test = None
    cfg_base = None

