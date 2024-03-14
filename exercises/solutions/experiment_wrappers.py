import ray
from ray import tune
from ray.tune.registry import register_env

from inventory_env_hard.env_creator import inventory_env_hard_creator


register_env("inventory_env_hard", inventory_env_hard_creator)


if __name__ == "__main__":
    # I have 28 CPU cores on my machine. I restricted ray to use only 50% of the full capacity.
    ray.init(num_cpus=14)

    tune.run("PPO",
             config={"env": "inventory_env_hard",
                     "env_config": {
                        "obs_filter": tune.grid_search(["my_normalize", "gym_normalize", None]),
                        "reward_filter": tune.grid_search(["my_scale_rewards", "gym_scale_rewards", None]),
                        },
                     "evaluation_config": {
                         "env_config": {
                             "reward_filter": None,
                             "obs_filter": tune.sample_from(lambda spec: spec.config.env_config.obs_filter),
                             }
                         },
                     "num_workers": 1,
                     "evaluation_interval": 500,
                     "evaluation_num_episodes": 10000,
                     "always_attach_evaluation_results": True,
                     },
             local_dir="experiment_results",
             name="experiment_wrappers",
             checkpoint_freq=500,
             # Using a stopping criteria so that Ray Tune can queue experiments
             stop={"agent_timesteps_total": 20000000}
             )
