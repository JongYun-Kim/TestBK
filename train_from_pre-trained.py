import ray
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo import PPOTrainer
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
    save_name = "myenv_{}_agent{}_task{}".format(rl_algo, num_uav, num_task)
    restore_path = "../../ray_results/PPOTrainer_myenv_3_8_2022-09-11_22-50-15bjqic3gj\checkpoint_009990/checkpoint-9990"

    # Get configuration dictionary for the trainer
    cfg = {"env": "myenv_3_8",
           "framework": "torch",
           # Get your GPU if needed and workers as much as you need
           "num_gpus": 1,
           "num_workers": 10,
           # "num_gpus_per_worker": 0.1,
           # Train batch size: if an episode is longer than 200, consider change the fragment size
           "train_batch_size": 8000,
           # All model-related settings go into this sub-dict.
           "model": {
                     # By default, the MODEL_DEFAULTS dict above will be used.
                     # Change individual keys in that dict by overriding them, e.g.
                     "fcnet_hiddens": [256, 256, 256, 256],  # [64,64][1024,1024][128,32][32,32] //[512, 512, 512],
                     "fcnet_activation": "relu",  # Check how many relu nodes die
                    },
           }

    # Define your trainable object
    trainer = PPOTrainer(config=cfg)
    # Restore the agent
    trainer.restore(restore_path)

    for i in range(9991, 11001):
        # Perform on iteration fo training the policy with the algo.
        result = trainer.train()
        print(pretty_print(result))

        if i % 10 == 0:
            checkpoint = trainer.save()
            print("Checkpoint has been shaved at ", checkpoint)

        print("Current iteration: ", i)
        print()
