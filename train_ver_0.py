import ray
from ray import tune
# from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env
from TAProtoEnv_stationary import TAProtoEnvStationary

"""
Current ray dependency: ray 1.13.0 
Ray 2.0.0 or higher may not work with the script below, due to the amended APIs.
"""

if __name__ == "__main__":
    # Initialize ray
    ray.init()

    # Simulation (env) params
    num_uav = 3
    num_task = 8

    # Register your environment
    register_env("myenv_3_8", lambda _: TAProtoEnvStationary(num_uav, num_task))

    # Configuration settings
    rl_algo = "PPO"
    # rl_algo = "IMPALA"
    save_name = "myenv_{}_agent{}_task{}".format(rl_algo, num_uav, num_task)

    tune.run(rl_algo,
             name=save_name,
             stop={"training_iteration": 3000},  # Never set this if you intend to continue the training later on.
             checkpoint_freq=10,
             # resume="LOCAL",
             config={"env": "myenv_3_8",
                     "framework": "torch",
                     # Get your GPU if needed and workers as much as you need
                     "num_gpus": 1,
                     "num_workers": 6,
                     # Train batch size: if an episode is longer than 200, consider change the fragment size
                     "train_batch_size": 8000,
                     # All model-related settings go into this sub-dict.
                     "model": {
                               # By default, the MODEL_DEFAULTS dict above will be used.
                               # Change individual keys in that dict by overriding them, e.g.
                               "fcnet_hiddens": [256, 256, 256, 256],  # [64,64][1024,1024][128,32][32,32] //[512, 512, 512],
                               "fcnet_activation": "relu",  # Chcek how many relu nodes die
                              },
                     },
             )
