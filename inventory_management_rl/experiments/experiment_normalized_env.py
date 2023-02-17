import ray
from ray import tune
from ray.tune.registry import register_env

from inventory_env.env_creator import inventory_env_creator


register_env("inventory_env", inventory_env_creator)


if __name__ == "__main__":
    ray.init()

    tune.run("PPO",
             # This works because the string "CartPole-v1" is registered in tune's environment registry
             config={"env": "inventory_env",
                     "env_config": {
                        # possible values: "my_normalize", "gym_normalize"
                        # Ray Tune Search Spaces: tune.grid_search()
                        "obs_filter": tune.grid_search(["my_normalize", "gym_normalize"]),
                        # possible values: "my_scale_rewards", "gym_scale_rewards"
                        "reward_filter": tune.grid_search(["my_scale_rewards", "gym_scale_rewards"])
                        },
                     # Might want to apply different wrappers during evaluation
                     # We will not apply any reward wrapper during evaluation
                     # Use the exact same observation and action wrappers in evaluation (as during training)
                     "evaluation_config": {
                         "reward_filter": None,
                         # Conditional Search Space: tune.sample_from()
                         "obs_filter": tune.sample_from(lambda spec: spec.config.env_config.obs_filter)
                         }
                     },
             local_dir="experiment_results"
             )
