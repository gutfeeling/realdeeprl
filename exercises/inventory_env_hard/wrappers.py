import gym
from gym.spaces import Box
import numpy as np


class MakeHard(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_goodwill_penalty_per_unit = 10
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        self.observation_space = Box(
            low=np.append(obs_low, 0),
            high=np.append(obs_high, self.max_goodwill_penalty_per_unit)
        )
        self.obs_dim = self.env.obs_dim + 1

    def reset(self):
        obs = self.env.reset()
        self.goodwill_penalty_per_unit = self.env.rng.uniform() * self.max_goodwill_penalty_per_unit
        return np.append(obs, self.goodwill_penalty_per_unit)

    def step(self, action):
        on_hand_inventory = self.env.current_obs[0]
        obs, r, done, info = self.env.step(action)
        demand = info["demand"]
        goodwill_penalty = - self.goodwill_penalty_per_unit * max(0, demand - on_hand_inventory)
        return np.append(obs, self.goodwill_penalty_per_unit), r + goodwill_penalty, done, info


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
        avg_goodwill_penalty_per_unit = self.env.max_goodwill_penalty_per_unit / 2
        avg_unmet_demand = self.env.max_mean_daily_demand / 4
        avg_high_scale = avg_unit_selling_price * avg_num_items_sold_per_day
        avg_low_scale = - (avg_unit_buying_price * avg_num_items_bought_per_day +
                           avg_daily_holding_cost_per_unit * avg_num_items_held_per_day +
                           avg_goodwill_penalty_per_unit * avg_unmet_demand
                           )
        # Linear transformation that maps avg_high_scale and avg_low_scale to +1 and -1
        mid = (avg_high_scale + avg_low_scale) / 2
        linearly_mapped_reward = 2 * (reward - mid) / (avg_high_scale - avg_low_scale)
        return np.arctan(linearly_mapped_reward) / np.arctan(1)
