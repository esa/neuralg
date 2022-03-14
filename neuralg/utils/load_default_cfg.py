from json import load
import os
import toml
from dotmap import DotMap


def load_default_cfg():
    """Loads the default toml config file from the cfg folder."""
    # Right now I have only gotten it to work on my local device
    path = "/Users/toveagren/ESA/NLAAP/neuralg/utils/default_config.toml"
    # This is not working: Just says it cannot find the file os.path.join(os.path.dirname(__file__), "default_config.toml")
    with open(path) as cfg:
        return DotMap(toml.load(cfg))


def load_train_config():
    return None

