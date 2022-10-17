config={"env": "NSenv",
                     "framework": "torch",
                     # Get your GPU if needed and workers as much as you need
                     "num_gpus": 1,
                     "num_workers": 6,
                     # "rollout_fragment_length": 1250,
                     # Train batch size: if an episode is longer than 200, consider change the fragment size
                     "train_batch_size": 6000,
                     "vf_clip_param": 20,
                     # All model-related settings go into this sub-dict.
                     "model": {
                               # By default, the MODEL_DEFAULTS dict above will be used.
                               # Change individual keys in that dict by overriding them, e.g.
                               "fcnet_hiddens": [512, 512, 256],  # [64,64][1024,1024][128,32][32,32] //[512, 512, 512],
                               "fcnet_activation": "relu",  # Check how many relu nodes die
                               # == LSTM ==
                               # Whether to wrap the model with an LSTM.
                               "use_lstm": True,  # False
                               # Max seq len for training the LSTM, defaults to 20.
                               "max_seq_len": 4,  # 20
                               # Size of the LSTM cell.
                               "lstm_cell_size": 256,  # 256
                               # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
                               "lstm_use_prev_action": True,  # False
                               # Whether to feed r_{t-1} to LSTM.
                               "lstm_use_prev_reward": False,  # False
                               # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
                               "_time_major": False,  # False
                              },
        
                     }

import ray
from ray import tune
from ray.tune.registry import register_env
from TARL_moving_targets import TAEnvMovingTasks
register_env("NSenv", lambda _: TAEnvMovingTasks(2,30,10))
from ray.rllib.agents.ppo.ppo import PPOTrainer
agent = PPOTrainer(config=config)
restore_path = "../../ray_results/PPOa2t30v10/PPO_NSenv_2878d_00000_0_2022-10-14_11-30-01/checkpoint_001000/checkpoint-1000"
agent.restore(restore_path)
policy = agent.get_policy()
model = policy.model
