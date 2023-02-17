from gym.wrappers.normalize import NormalizeReward

from inventory_env.inventory_env import InventoryEnv
from inventory_env.wrappers import MyNormalizeObservation, MyScaleReward, GymNormalizeObservation


def inventory_env_creator(config):
    obs_filter = config.pop("obs_filter", None)
    reward_filter = config.pop("reward_filter", None)
    env = InventoryEnv()
    if obs_filter is not None:
        if obs_filter == "my_normalize":
            env = MyNormalizeObservation(env)
        elif obs_filter == "gym_normalize":
            env = GymNormalizeObservation(env)
    if reward_filter is not None:
        if reward_filter == "my_scale_rewards":
            env = MyScaleReward(env)
        elif reward_filter == "gym_scale_rewards":
            env = NormalizeReward(env)
    return env
