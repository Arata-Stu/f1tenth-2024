import os
from typing import Tuple

import math
from omegaconf import DictConfig, open_dict


def dynamically_modify_config(config: DictConfig):
    with open_dict(config):
        config.reward.steer_range = config.vehicle.steer_range
        config.reward.speed_range = config.vehicle.speed_range

        