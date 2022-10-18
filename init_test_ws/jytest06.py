import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.mpe import simple_tag_v2
import supersuit as ss
import time


def env_creator(args):
    # return PettingZooEnv(waterworld_v3.env(n_pursuers=10, n_evaders=10))
    env_o = simple_tag_v2.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=400, continuous_actions=False)
    env_o = ss.pad_observations_v0(env_o)
    return env_o


def policy_mapping_fn(agent_id):
    if agent_id.startswith("adversary"):
        return "adversary_policy"
    elif agent_id.startswith("good"):
        return "good_policy"
    else:
        print("THERE IS NO POLICY MAPPED ADEQUATELY")
        return "UNMAPPED_policy"


# Initialize ray.
ray.init()
# Register your environment.
register_env("mpe", lambda args: PettingZooEnv(env_creator(args)))
# Create the environment
env = PettingZooEnv(env_creator({}))

# Define your observation and action spaces.
obs_space = env.observation_space
act_space = env.action_space

# Define your policies.
policies = {"adversary_policy": (None, obs_space, act_space, {}),
            "good_policy": (None, obs_space, act_space, {})
            }
# Get your policy ids.
policy_ids = list(policies.keys())

# Get the learning agent.
PPOagent = PPOTrainer(config={  # Enviroment specific
                            "env": "mpe",
                            # General
                            "framework": "torch",
                            # "callbacks": MyCallbacks,
                            "num_gpus": 1,
                            "num_workers": 0,
                            # "object_store_memory": 10 ** 9,
                            # Method specific
                            "multiagent": {
                                           "policies": policies,
                                           # "policies_to_train": ["shared_policy"],
                                           "policy_mapping_fn": policy_mapping_fn,
                                           # "policy_mapping_fn": (lambda agent_id: "shared_policy_1"),
                                           # "policy_mapping_fn": ray.tune.function(
                                           #     lambda i: policy_ids[i]
                                           # ),
                                           },
                            }
                      )

# PPOagent.restore("../../ray_results/PPO_mpe_pray1_pred3_torch_trial_0/PPO_mpe_d91f4_00000_0_2022-08-03_02-59-03/checkpoint_000200/checkpoint-200")
PPOagent.restore("../../ray_results/PPO_mpe_pray1_pred3_torch_trial_1/PPO_mpe_250a2_00000_0_2022-08-03_06-14-26/checkpoint_000700/checkpoint-700")

# Reset the environment before using the environment for the trained agent.
env.reset()

# Get your agent trained implemented in the environment.
for agent in env.env.agent_iter():  # Agent by agent in the order until the termination condition met
    # Get observation for the agent
    observation, reward, done, info = env.env.last()
    # Get your action based on the corresponding policy unless it's been done.
    if done:
        action = None
    else:
        action, _, _ = PPOagent.get_policy(policy_mapping_fn(agent)).compute_single_action(observation)
    # Get your step forward for the agent.
    env.env.step(action)
    # Render the env only when all the agent has updated there state.
    if agent == 'adversary_0':
        env.env.render()
        time.sleep(0.12)
# Take a break to appreciate the last moment
time.sleep(3)
# before closing the environment visualized
env.close()
