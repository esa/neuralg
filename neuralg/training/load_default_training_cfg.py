import os
import toml
from dotmap import DotMap


def load_default_training_cfg():
    """Loads the default toml training config file from the training folder"""
    path = os.path.join(os.path.dirname(__file__), "default_training_config.toml")
    with open(path) as cfg:
        return DotMap(toml.load(cfg))

