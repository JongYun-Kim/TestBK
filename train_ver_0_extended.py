import ray
from ray import tune
# from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env
from TAProtoEnv_stationary import TAProtoEnvStationary
from TARL_moving_targets import TAEnvMovingTasks
from TARL_varying_speed_tasks import TAEnvMovingTasksInEpisode
from myICM import ReactiveExploration
from datetime import datetime

"""
Current ray dependency: ray 1.13.0 (or 1.10.0)
Ray 2.0.0 or higher may not work with the script below, due to the amended APIs.
"""


# def trial_name_id(trial):
#     return f"{trial.trainable_name}_{trial.trial_id}"


def trial_dir_name_creator(trial):
    env_name = trial.config["env"]
    now = datetime.now()
    time_str = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
    trial_name = env_name + "_" + time_str
    return trial_name


if __name__ == "__main__":
    # Initialize ray
    ray.init()

    # Simulation (env) params
    num_uav = 2
    num_task = 30
    task_vel = 10
    acp = 10

    # Register your environment
    # register_env("myenv", lambda _: TAProtoEnvStationary(num_uav, num_task))
    # register_env("NSenv", lambda _: TAEnvMovingTasks(num_uav, num_task, task_vel))
    register_env("NSenv2", lambda _: TAEnvMovingTasksInEpisode(num_uav, num_task, task_vel, acp))
    target_env = "NSenv2"

    # Configuration settings
    rl_algo = "PPO"
    # rl_algo = "IMPALA"
    save_name = "{}a{}t{}v{}acp{}".format(rl_algo, num_uav, num_task, task_vel, acp)
    # save_name = "dummy_runs"

    tune.run(rl_algo,
             trial_dirname_creator=trial_dir_name_creator,
             name=save_name,
             # stop={"training_iteration": 3000},  # Never set this if you intend to resume the training later on.
             checkpoint_freq=20,
             # resume=True,
             config={"env": target_env,
                     "framework": "torch",
                     # Get your GPU if needed and workers as much as you need
                     "num_gpus": 1,
                     "num_workers": 8,
                     # "rollout_fragment_length": 1250,
                     # Train batch size: if an episode is longer than 200, consider change the fragment size
                     "train_batch_size": 8000,
                     "vf_clip_param": 20,
                     "vf_loss_coeff": 0.00001,  # used to be 1.0. For scaling of vf loss wrt policy loss
                     # All model-related settings go into this sub-dict.
                     "model": {
                               # By default, the MODEL_DEFAULTS dict above will be used.
                               # Change individual keys in that dict by overriding them, e.g.
                               "fcnet_hiddens": [512, 512, 256, 256],
                               "fcnet_activation": "relu",  # Check how many relu nodes die
                               # # == LSTM ==
                               # # Whether to wrap the model with an LSTM.
                               # "use_lstm": True,  # False
                               # # Max seq len for training the LSTM, defaults to 20.
                               # "max_seq_len": 4,  # 20
                               # # Size of the LSTM cell.
                               # "lstm_cell_size": 256,  # 256
                               # # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
                               # "lstm_use_prev_action": True,  # False
                               # # Whether to feed r_{t-1} to LSTM.
                               # "lstm_use_prev_reward": False,  # False
                               # # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
                               # # "_time_major": False,  # False
                              },
                     # "exploration_config": {"type": ReactiveExploration,
                     #                        # <- Use the Curiosity module for exploring.
                     #                        "eta": 1.3,  # Weight for intrinsic rewards before being added to extrinsic ones.
                     #                        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
                     #                        "feature_dim": 64,  # Dimensionality of the generated feature vectors.
                     #                        # Setup of the feature net (used to encode observations into feature (latent) vectors).
                     #                        "feature_net_config": {"fcnet_hiddens": [256],
                     #                                               "fcnet_activation": "relu",
                     #                                               },
                     #                        "inverse_net_hiddens": [64, 64],  # Hidden layers of the "inverse" model.
                     #                        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                     #                        "forward_net_hiddens": [64, 64],  # Hidden layers of the "forward" model.
                     #                        "forward_net_activation": "relu",  # Activation of the "forward" model.
                     #                        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                     #                        # Specify, which exploration sub-type to use (usually, the algo's "default"
                     #                        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                     #                        "sub_exploration": {"type": "StochasticSampling",
                     #                                            },
                     #                        "eta_re": 0.00003,  # Scaling factor of RE reward
                     #                        "loss_coeff": 0.00003,  # Scaling factor for loss in RE_net
                     #                        "lr_re": 0.001,
                     #                        "re_net_hiddens": [256, 64],
                     #                        "re_net_activation": "relu",
                     #                        },
                     },
             )
