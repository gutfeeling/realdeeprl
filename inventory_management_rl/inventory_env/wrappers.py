import gym
from gym.wrappers.normalize import NormalizeObservation
from gym.spaces import Box
import numpy as np


class MyNormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=np.zeros((self.env.obs_dim,)),
                                     high=np.ones((self.env.obs_dim,))
                                     )

    def observation(self, obs):
        return obs / self.env.observation_space.high


class MyScaleReward(gym.RewardWrapper):
    def reward(self, reward):
        """
        :param reward: The reward from the original environment
        :return: Post-processed reward
        RewardWrapper will automatically take care of applying this reward post-processing
        in the wrapper's step() method
        """
        # Estimates of average parameter values
        avg_unit_selling_price = self.env.max_unit_selling_price / 2
        avg_num_items_bought_per_day = avg_num_items_sold_per_day = self.env.max_mean_daily_demand / 2
        avg_unit_buying_price = self.env.max_unit_selling_price / 4
        avg_daily_holding_cost_per_unit = self.env.max_daily_holding_cost_per_unit / 2
        avg_num_items_held_per_day = self.env.max_mean_daily_demand / 2
        avg_high_scale = avg_unit_buying_price * avg_num_items_sold_per_day
        avg_low_scale = - (avg_unit_buying_price * avg_num_items_bought_per_day +
                           avg_daily_holding_cost_per_unit * avg_num_items_held_per_day
                           )
        # Linear transformation that maps avg_high_scale and avg_low_scale to +1 and -1
        mid = (avg_high_scale + avg_low_scale) / 2
        linearly_mapped_reward = 2 * (reward - mid) / (avg_high_scale - avg_low_scale)
        return np.arctan(linearly_mapped_reward) / np.arctan(1)


class GymNormalizeObservation(NormalizeObservation):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.observation_space = Box(low=np.ones((self.env.obs_dim,)) * -np.inf,
                                     high=np.ones((self.env.obs_dim,)) * np.inf
                                     )
