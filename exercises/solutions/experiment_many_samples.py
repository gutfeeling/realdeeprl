import ray
from ray import tune
from ray.tune.registry import register_env

from inventory_env_hard.env_creator import inventory_env_hard_creator


register_env("inventory_env_hard", inventory_env_hard_creator)


if __name__ == "__main__":
    ray.init()

    tune.run("PPO",
             config={"env": "inventory_env_hard",
                     "env_config": {
                        "obs_filter": "my_normalize",
                        "reward_filter": "gym_scale_rewards",
                        },
                     "evaluation_config": {
                         "env_config": {
                             "reward_filter": None,
                             "obs_filter": "my_normalize",
                             }
                         },
                     "num_workers": 1,
                     "evaluation_interval": 500,
                     "evaluation_num_episodes": 10000,
                     "always_attach_evaluation_results": True,
                     },
             local_dir="experiment_results",
             name="experiment_many_samples",
             # ----- SOLUTION ----- #
             # Using the num_samples argument to run 4 copies of the same experiment
             num_samples=4,
             checkpoint_freq=500,
             )
