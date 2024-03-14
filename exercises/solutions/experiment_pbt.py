import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from inventory_env_hard.env_creator import inventory_env_hard_creator


register_env("inventory_env_hard", inventory_env_hard_creator)


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
    perturbation_interval=100,
    metric="evaluation/episode_reward_mean",
    mode="max",
    hyperparam_mutations={
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": [1, 5, 10, 20],
        "sgd_minibatch_size": [8, 16, 32, 64, 128, 256, 512],
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
             config={"env": "inventory_env_hard",
                     # The best wrapper combination that I found
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
                     # The best network size that I found (note how smaller was better in this case)
                     # The fcnet_activation is not explicitly defined because RLlib's default value turned out best
                     "model": {
                         "fcnet_hiddens": [64, 64],
                     },
                     "train_batch_size": 2048,
                     "sgd_minibatch_size": 128,
                     "evaluation_interval": 100,
                     "evaluation_num_episodes": 10000,
                     "always_attach_evaluation_results": True,
                     },
             local_dir="experiment_results",
             name="pbt",
             checkpoint_freq=100,
             # 4 trials will be competing in our population
             num_samples=4,
             scheduler=pbt,
             )
