import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from inventory_env.env_creator import inventory_env_creator


register_env("inventory_env", inventory_env_creator)


def postprocess(config):
    if config["entropy_coeff"] < 0.00000001:
        config["entropy_coeff"] = 0.00000001
    if config["entropy_coeff"] > 0.1:
        config["entropy_coeff"] = 0.1
    if config["train_batch_size"] < 2 * config["sgd_minibatch_size"]:
        config["train_batch_size"] = 2 * config["sgd_minibatch_size"]
    return config


pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    # perturbation interval should be an integer multiple of evaluation interval/checkpoint frequency
    perturbation_interval=100,
    metric="evaluation/episode_reward_mean",
    mode="max",
    hyperparam_mutations={
        # We should only tune hyperparams, not the network architecture (e.g. layer size, activation func etc.)
        # Otherwise, copying the weights (which happens during exploitation) wouldn't make sense
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        # mutations happen by choosing an adjacent value at random
        "num_sgd_iter": [1, 5, 10, 20],
        "sgd_minibatch_size": [8, 16, 32, 64, 128, 256, 512],
        # train batch size must be at least twice the minibatch size
        "train_batch_size": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        # mutations happen by multiplying current value with 0.8 or 1.2 (in case of cont. search spaces)
        "entropy_coeff": tune.loguniform(0.00000001, 0.1),
        "clip_param": [0.1, 0.2, 0.3, 0.4],
        "lambda": [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
    },
    custom_explore_fn=postprocess,
)


if __name__ == "__main__":
    ray.init()

    tune.run("PPO",
             config={"env": "inventory_env",
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
                     # custom_explore_fn is not called at the start, so we set fixed sensible values for minibatch size
                     # and train batch size
                     "train_batch_size": 2048,
                     "sgd_minibatch_size": 128,
                     "evaluation_interval": 100,
                     "evaluation_num_episodes": 10000,
                     # Necessary since PBT expects each result object to have the metric under optimization
                     "always_attach_evaluation_results": True,
                     },
             local_dir="experiment_results",
             name="pbt",
             checkpoint_freq=100,
             # 4 trials will be competing in our population
             num_samples=4,
             scheduler=pbt,
             )
