from omegaconf import DictConfig

from .reward import ProgressReward, PPReward
from .map import MapManager
from .env import F110_Wrapped
from f1tenth_gym.f110_env import F110Env

def build_reward(reward_conf:DictConfig, map_manager):
    if reward_conf.name == 'Progress':
        reward = ProgressReward(map_manager=map_manager, reward_gain=reward_conf.gain)
    elif reward_conf.name == 'TAL':
        reward_conf = PPReward(map_manager=map_manager,
                               steer_range=reward_conf.steer_range,
                               speed_range=reward_conf.speed_range,
                               max_theta=100,
                               steer_w=reward_conf.steer_w,
                               speed_w=reward_conf.speed_w,
                               alpha=reward_conf.alpha,
                               reward_gain=reward_conf.gain)
    else:
        NotImplementedError

    return reward


def build_env(config:DictConfig):
    map_manager = MapManager(map_name=config.map.name, raceline=config.map.line, delimiter=config.map.delimiter, speed=6.0 ,dir_path='./f1tenth_racetracks/')
    
    time_step = 0.025
    s_range = config.vehicle.steer_range
    v_range = config.vehicle.speed_range
    param = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -s_range, 's_max': s_range, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': v_range, 'width': 0.31, 'length': 0.58}
    env = F110Env(map=map_manager.map_path, map_ext='.png', timestep=time_step ,num_beams=config.vehicle.num_beams, beam_fov=config.vehicle.beam_fov, num_agents=1, params=param)
    env = F110Env(map=map_manager.map_path, map_ext=config.map_ext ,num_agents=1, num_beams = 1080)
    env = F110_Wrapped(env=env, map_manager=map_manager)

    return env
