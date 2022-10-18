"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import argparse
import gym
import os
import random
import ray
from ray.tune.registry import register_env
from ray.rllib.env.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v3
parser = argparse.ArgumentParser()
# Use torch for both policies.
parser.add_argument("--torch", action="store_true")
# Mix PPO=tf and DQN=torch if set.
parser.add_argument("--mixed-torch-tf", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1200)
parser.add_argument("--stop-reward", type=float, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--stop-iters", type=int, default=2000)


if __name__ == "__main__":

    args = parser.parse_args()

    ray.init()


    def env_creator(args):
        return PettingZooEnv(waterworld_v3.env(n_pursuers=10, n_evaders=10))

    env = env_creator({})
    register_env("waterworld", env_creator)

    obs_space = env.observation_space
    act_spc = env.action_space

    env.render(mode='human')

    # policies = {agent: (None, obs_space, act_spc, {}) for agent in env.agents}
    # policies = {"shared_policy_1": (None, obs_space, act_spc, {}),
    #             "shared_policy_2": (None, obs_space, act_spc, {})
    # policies = {"shared_policy_1": (None, obs_space, act_spc, {}),
    #             "shared_policy_2": (None, obs_space, act_spc, {})
    #                                 }
    policies = {"shared_1": (None, obs_space, act_spc, {})
                # "shared_2": (None, obs_space, act_spc, {})
                # "pursuer_5": (None, obs_space, act_spc, {})
                }


    # policy_ids = list(policies.keys())


    # def policy_mapping_fn(agent_id):
    #     if agent_id == "pursuer_0" or "pursuer_1" or "pursuer_2" or "pursuer_3" or "pursuer_4":
    #         # print("agent id = ", agent_id)
    #         return "shared_1"
    #     # elif agent_id == "pursuer_2" or "pursuer_3":
    #     #     # print("agent id = ", agent_id)
    #     #     return "shared_2"
    #     # elif agent_id == "pursuer_3":
    #     #     # print("agent id = ", agent_id)
    #     #     return "pursuer_3"
    #     else:
    #         # print("agent id = ", agent_id)
    #         return "shared_2"

    tune.run(
        "PPO",
        name="PPO shared n = 10 torch workers = 0 trial 5 new",
        stop={"episodes_total": 50000},
        checkpoint_freq=10,
        config={
            # Enviroment specific
            "env": "waterworld",
            # General
            "framework": "torch",
            # "callbacks": MyCallbacks,
            "num_gpus": 0,
            "num_workers": 0,
            # "object_store_memory": 10 ** 9,
            # Method specific
            "multiagent": {
                "policies": policies,
                # "policies_to_train": ["shared_policy"],
                # "policy_mapping_fn": policy_mapping_fn,
                "policy_mapping_fn": (lambda agent_id: "shared_1"),
            },
        },
    )