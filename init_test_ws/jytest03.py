"""
Simple tag with 2 policies (each for a type of agents) that use self play of the same policy half of the time
and learn the other half

"""

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.mpe import simple_tag_v2
from supersuit import pad_observations_v0
from pettingzoo.sisl import waterworld_v3
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import argparse
import numpy as np
import random
import ray
from ray.tune.registry import register_env
from ray.rllib.env.pettingzoo_env import PettingZooEnv
import copy

# start_pps = 10  # the number of iterations there is no weight sharing (in the beginning)
# T = 5  # the period of the weight sharing
M = 10  # Menagerie size
# men = {}
# men_rewards = {}
# men_start = 1
# nan_true = 0


class MyCallbacks(DefaultCallbacks):

    def __init__(self):
        super(MyCallbacks, self).__init__()
        self.nan_counter = 0
        self.men1 = []
        self.men_rewards1 = []
        self.men2 = []
        self.men_rewards2 = []

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))
        # for i in range(3):  # train iter
        #     result = trainer.train()
        i = result['training_iteration']    # starts from 1
        # print("training iteration", i)
        max_reward = result['episode_reward_mean']
        # the "shared_policy_1" is the only agent being trained
        if np.isnan(result['episode_reward_mean']):
            # global men_start, nan_true
            # men_start = i
            self.nan_counter += 1    # flag for nana in the beginning
            pass
        elif i <= M + self.nan_counter:
            # menagerie initialisation
            self.men1.append(copy.deepcopy(trainer.get_policy("shared_policy_1").get_weights()))
            self.men_rewards1.append(result['episode_reward_mean'])

            """ the result is the result.json that has this policy reward mean in a nested dict """
            #
            # self.men2.append(copy.deepcopy(trainer.get_policy("shared_policy_2").get_weights()))
            # self.men_rewards2.append(result['policy_reward_mean']['shared_policy_2'])

            # print("iteartion", i)
            # print("counter", i - 1 - self.nan_counter)
            # trainer.set_weights({"shared_policy_1": self.men1[i-2-self.nan_counter]})
            #
            # trainer.set_weights({"shared_policy_2": self.men2[i-2-self.nan_counter]})
            #
            # weights = ray.put(trainer.workers.local_worker().save())
            # trainer.workers.foreach_worker(
            #     lambda w: w.restore(ray.get(weights))
            # )

        elif i % 2 == 0 and i > M + self.nan_counter:
            # key_min = min(men_rewards, key=lambda x: men_rewards[x])
            key_min1 = np.argmin(self.men_rewards1)

            # key_min2 = np.argmin(self.men_rewards2)

            # print("key_min1 = ", key_min1)
            # print("rewards1", self.men_rewards1)
            # print("key_min2 = ", key_min2)
            # print("rewards2", self.men_rewards2)

            current_reward1 = result['policy_reward_mean']['shared_policy_1']

            # current_reward2 = result['policy_reward_mean']['shared_policy_2']

            # print("current reward1 = ", current_reward1, "min reward1 =", self.men_rewards1[key_min1])
            # print("current reward2 = ", current_reward2, "min reward2 =", self.men_rewards2[key_min2])

            if current_reward1 > self.men_rewards1[key_min1]:
                self.men1.pop(key_min1)
                self.men1.append(copy.deepcopy(trainer.get_policy("shared_policy_1").get_weights()))
                self.men_rewards1.pop(key_min1)
                self.men_rewards1.append(current_reward1)
                print("menagerie updated")
                # select one policy randomly. The menagerie length must exlude potential nans in the beginning

                sel = random.randint(0, M - 1)

                trainer.set_weights(
                    {"shared_policy_1": self.men1[sel]  # weights or values from "policy_1" with "policy_0" keys
                     })

                weights = ray.put(trainer.workers.local_worker().save())
                trainer.workers.foreach_worker(
                    lambda w: w.restore(ray.get(weights))
                )

                # you can mutate the result dict to add new fields to return
                # print("rewards1", self.men_rewards1)
                # current_reward = result['episode_reward_mean']
            else:

                # print(key_min)
                # print("rewards1", self.men_rewards1)
                # current_reward1 = result['policy_reward_mean']['shared_policy_1']
                # print("current reward1", current_reward1)

                sel = random.randint(0, M - 1)

                # sel = random.randint(1, M)

                trainer.set_weights(
                    {"shared_policy_1": self.men1[sel]  # weights or values from "policy_1" with "policy_0" keys
                     })

                weights = ray.put(trainer.workers.local_worker().save())
                trainer.workers.foreach_worker(
                    lambda w: w.restore(ray.get(weights))
                )

                # you can mutate the result dict to add new fields to return
                # print("menagerie remained the same")
                # print("rewards1", self.men_rewards1)
                ## current_reward = result['episode_reward_mean']
                # print("current reward1 = ", current_/reward1, "min reward1 =", self.men_rewards1[key_min1])

            ################################### policy 2  ##########################################
            #
            # if current_reward2 > self.men_rewards2[key_min2]:
            #     self.men2.pop(key_min2)
            #     self.men2.append(copy.deepcopy(trainer.get_policy("shared_policy_2").get_weights()))
            #     self.men_rewards2.pop(key_min2)
            #     self.men_rewards2.append(current_reward2)
            #     print("menagerie updated")
            #     # select one policy randomly. The menagerie length must exlude potential nans in the beginning
            #
            #     sel = random.randint(0, M - 1)
            #
            #     trainer.set_weights(
            #         {"shared_policy_2": self.men2[sel]  # weights or values from "policy_1" with "policy_0" keys
            #          })
            #
            #     weights = ray.put(trainer.workers.local_worker().save())
            #     trainer.workers.foreach_worker(
            #         lambda w: w.restore(ray.get(weights))
            #     )
            #
            #     # you can mutate the result dict to add new fields to return
            #     # print("rewards2", self.men_rewards2)
            #     # current_reward = result['episode_reward_mean']
            # else:
            #
            #     # print(key_min)
            #     # print("rewards2", self.men_rewards2)
            #     # current_reward2 = result['policy_reward_mean']['shared_policy_2']
            #     # print("current reward2", current_reward2)
            #
            #     sel = random.randint(0, M - 1)
            #
            #     # sel = random.randint(1, M)
            #
            #     trainer.set_weights(
            #         {"shared_policy_2": self.men2[sel]  # weights or values from "policy_1" with "policy_0" keys
            #          })
            #
            #     weights = ray.put(trainer.workers.local_worker().save())
            #     trainer.workers.foreach_worker(
            #         lambda w: w.restore(ray.get(weights))
            #     )
            #
            #     # you can mutate the result dict to add new fields to return
            #     # print("menagerie remained the same")
            #     # print("rewards2", self.men_rewards2)
            #     ## current_reward = result['episode_reward_mean']
            #     # print("current reward2 = ", current_reward2, "min reward2 =", self.men_rewards2[key_min2])
        else:
            pass

        result["callback_ok"] = True


if __name__ == "__main__":

    # args = parser.parse_args()

    ray.init()

    def env_creator(args):
        return PettingZooEnv(waterworld_v3.env(n_pursuers=10, n_evaders=10))

    env = env_creator({})
    register_env("waterworld", env_creator)

    obs_space = env.observation_space
    act_spc = env.action_space

    # policies = {agent: (None, obs_space, act_spc, {}) for agent in env.agents}
    policies = {"shared_policy_1": (None, obs_space, act_spc, {}),
                # "shared_policy_2": (None, obs_space, act_spc, {})
                }

    # policy_ids = list(policies.keys())


    # def policy_mapping_fn(agent_id):
    #     if agent_id == "pursuer_0" or "pursuer_1" or "pursuer_2" or "pursuer_3" or "pursuer_4":
    #         return "shared_policy_1"
    #     else:
    #         return "shared_policy_2"

    policy_ids = list(policies.keys())

    tune.run(
        "PPO",
        name="PPO waterworld max reward mixed M=10 n=10 torch trial 1",
        # stop={"episodes_total": 50000},
        checkpoint_freq=10,
        config={
            # Enviroment specific
            "env": "waterworld",
            # General
            "framework": "torch",
            "callbacks": MyCallbacks,
            "num_gpus": 0,
            "num_workers": 0,
            # "object_store_memory": 10 ** 9,
            # Method specific
            "multiagent": {
                "policies": policies,
                # "policies_to_train": ["shared_policy"],
                # "policy_mapping_fn": policy_mapping_fn,
                "policy_mapping_fn": (lambda agent_id: "shared_policy_1"),
            },
        },
    )