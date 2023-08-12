import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env

from inventory_env.env_creator import inventory_env_creator


register_env("inventory_env", inventory_env_creator)


def get_train_batch_size(spec):
    minibatch_size = spec.config.sgd_minibatch_size
    train_batch_size = np.random.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    if train_batch_size < 2 * minibatch_size:
        train_batch_size = 2 * minibatch_size
    return train_batch_size


if __name__ == "__main__":
    ray.init()

    tune.run("PPO",
             config={"env": "inventory_env",
                     "env_config": {
                         "obs_filter": "my_normalize",
                         "reward_filter": "gym_scale_rewards"
                     },
                     "evaluation_config": {
                         "env_config": {
                             "reward_filter": None,
                             "obs_filter": "my_normalize",
                         }
                     },
                     "model": {
                         "fcnet_hiddens": tune.grid_search([[256, 256], [64, 64]]),
                         "fcnet_activation": tune.grid_search(["tanh", "relu"]),
                     },
                     "sgd_minibatch_size": tune.grid_search([8, 16, 32, 64, 128, 256, 512]),
                     "train_batch_size": tune.sample_from(get_train_batch_size),
                     "entropy_coeff": tune.loguniform(0.00000001, 0.1),
                     "evaluation_interval": 500,
                     "evaluation_num_episodes": 10000,
                     "always_attach_evaluation_results": True,
                     },
             local_dir="experiment_results",
             name="hyperparameter_grid_search",
             checkpoint_freq=500,
             num_samples=4,
             )
