import ray
from ray import tune
from ray.tune.registry import register_env

from inventory_env.env_creator import inventory_env_creator


register_env("inventory_env", inventory_env_creator)


if __name__ == "__main__":
    ray.init()

    tune.run("PPO",
             config={"env": "inventory_env",
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
                     # Necessary when using tune.ExperimentAnalysis with evaluation related metrics
                     "always_attach_evaluation_results": True,
                     },
             local_dir="experiment_results",
             name="experiment_wrappers",
             checkpoint_freq=500
             )
