# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import PIL
from torch.utils.collect_env import get_pretty_env_info


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_pil_version()
    return env_str


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)
