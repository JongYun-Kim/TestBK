import ray
# from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env
from TAProtoEnv_stationary import TAProtoEnvStationary
import numpy as np
import matplotlib.pyplot as plt


def coordify_obs(obs_in, num_r, num_t):
    task_pos = obs_in[0:2*num_t].copy()
    task_pos = task_pos.reshape((2, num_t))
    task_pos = task_pos.T
    uav_pos = obs_in[2*num_t:2*(num_r+num_t)].copy()
    uav_pos = uav_pos.reshape((2, -1))
    uav_pos = uav_pos.T
    task_dones = obs_in[2*(num_r+num_t):2*(num_r+num_t)+num_t].copy()
    return uav_pos, task_pos, task_dones


def get_agents():
    # Configuration settings
    """
    # Agent1: 32-32,           tanh, cp-1000, 08-30-20-30
    # Agent2: 32-32-32-32,     relu, cp-1000, 08-31-19-47
    # Agent3: 256-128-32,      relu, cp-4280, 09-02-18-22  (128-64-16 도 비슷한 경향으로 학습함, 약간만 mean낮게, min높게)
    # Agent4: 682-682,         relu, cp-3210, 09-03-17-03
    # Agent5: 256-256-256,     relu, cp-1500, 09-01-18-06
    # Agent6: 256-256-256-265, relu, cp-3500?, 09-04-12-05
    """

    exploration = False

    cfg1 = {"env": "myenv",
            "framework": "torch",
            "model": {"fcnet_hiddens": [32, 32]},
            "explore": exploration,
            }
    cfg2 = {"env": "myenv",
            "framework": "torch",
            "model": {"fcnet_activation": "relu",
                      "fcnet_hiddens": [32, 32, 32, 32],
                      },
            "explore": exploration,
            }
    cfg3 = {"env": "myenv",
            "framework": "torch",
            "model": {"fcnet_activation": "relu",
                      "fcnet_hiddens": [256, 128, 32],
                      },
            "explore": exploration,
            # "train_batch_size": 8000,
            }
    cfg4 = {"env": "myenv",
            "framework": "torch",
            "model": {"fcnet_activation": "relu",
                      "fcnet_hiddens": [682, 682],
                      },
            "explore": exploration,
            # "train_batch_size": 8000,
            }
    cfg5 = {"env": "myenv",
            "framework": "torch",
            "model": {"fcnet_activation": "relu",
                      "fcnet_hiddens": [256, 256, 256],
                      },
            "explore": exploration,
            }
    cfg6 = {"env": "myenv",
            "framework": "torch",
            "model": {"fcnet_hiddens": [256, 256, 256, 256],
                      "fcnet_activation": "relu",
                      },
            "explore": exploration,
            # "train_batch_size": 8000,
            }
    # Get your learning agents
    agent1 = PPOTrainer(config=cfg1)
    agent2 = PPOTrainer(config=cfg2)
    agent3 = PPOTrainer(config=cfg3)
    agent4 = PPOTrainer(config=cfg4)
    agent5 = PPOTrainer(config=cfg5)
    agent6 = PPOTrainer(config=cfg6)

    agent1.restore("../../ray_results/myenv_PPO_agent2_task3/PPO_myenv_226da_00000_0_2022-08-30_20-30-07/checkpoint_001000/checkpoint-1000")
    agent2.restore("../../ray_results/myenv_PPO_agent2_task3/PPO_myenv_5bc24_00000_0_2022-08-31_19-47-29/checkpoint_001000/checkpoint-1000")
    agent3.restore("../../ray_results/myenv_PPO_agent2_task3/PPO_myenv_cfefa_00000_0_2022-09-02_18-22-33/checkpoint_004280/checkpoint-4280")
    agent4.restore("../../ray_results/myenv_PPO_agent2_task3/PPO_myenv_ff8f7_00000_0_2022-09-03_17-03-58/checkpoint_003210/checkpoint-3210")
    agent5.restore("../../ray_results/myenv_PPO_agent2_task3_256-256-256/PPO_myenv_6a81a_00000_0_2022-09-01_18-06-30/checkpoint_001500/checkpoint-1500")
    agent6.restore("../../ray_results/myenv_PPO_agent2_task3_256_256_256_256/PPO_myenv_69a63_00000_0_2022-09-04_12-05-00/checkpoint_002250/checkpoint-2250")

    agent_list_output = [agent1, agent2, agent3, agent4, agent5, agent6]

    return agent_list_output


class StoreMyResults:
    def __init__(self, num_exp=2048, num_agents=10):
        if num_exp == 2048:
            print("StoreMyResults __init__: num_exp may have not been given. Check if you did so")
        self.actions = np.zeros([num_exp, num_agents, 3, 2])  # 32: step, num_uav
        self.optimal_actions = np.zeros([num_exp, 3, 2])
        self.reward_sum = np.zeros([num_exp, num_agents, 1])
        self.optimal_reward_sum = np.zeros([num_exp,1])
        self.reward_sum_list = np.zeros([num_exp, 24])
        self.worst_reward_sum = np.zeros([num_exp,1])

    def say_hello(self):
        print("Hello World")
        return True


if __name__ == "__main__":
    # Initialize ray
    ray.init()

    # How many time do you want to run the test?
    num_test = 1000
    Results = StoreMyResults(num_test, 6)

    # Simulation params
    num_uav = 2
    num_task = 3

    # Register your environment
    register_env("myenv", lambda _: TAProtoEnvStationary(num_uav, num_task))

    # Get agents of your interest
    agent_list = get_agents()

    # Define the environment
    env = TAProtoEnvStationary(num_uav, num_task)

    # Run the experiment
    for episode in range(num_test):
        print("\n--------------------------------------------------------------------------")
        print(f"----------------------------- Episode {episode+1} ----------------------------------")
        '''
        (1) 리셋하기: env 등
        (2) 에이전트 돌리기
        (3) 에이전트 결과 저장하기 
        (4) 최적값구하기
        (5) 최적 결과 저장하기
        '''
        # (1) Reset the environment and get an initial observation
        env.reset()  # initial reset (env (uav,task) generation)

        # (2) Test run the agents in the loop
        for agent_idx, Agent in enumerate(agent_list):
            obs = env.reset(True)  # Go back to the initial state!
            done = False
            step = 0
            reward = 0
            while not done:
                action = Agent.compute_single_action(obs)
                obs, reward, done, _ = env.step(action)
                # (3) Save the results of the trained agents
                Results.actions[episode, agent_idx, step, :] = action
                Results.reward_sum[episode, agent_idx, 0] += reward
                step += 1

            # (4) Get the optimal results in the given environment
            env.reset(True)  # Go back to the initial state!
            optimal_episode_actions, optimal_episode_reward, reward_sum_set = env.get_optimum()
            # (5) Save the optimal results
            Results.optimal_actions[episode, :, :] = optimal_episode_actions
            Results.optimal_reward_sum[episode, 0] = optimal_episode_reward
            Results.reward_sum_list[episode, :] = reward_sum_set
            Results.worst_reward_sum[episode, 0] = reward_sum_set.min()

    # Close your environment
    env.close()
    ray.shutdown()

    # Visualize the results
    x = np.arange(1, num_test+1, 1)  # test numbers
    rewards_plot_data = np.zeros([len(agent_list)+2, num_test, 1])  # optimal이랑 worst 둘다 계산

    # Optimal and worst data
    y_data_optimal = Results.optimal_reward_sum.squeeze().copy()
    idx = np.argsort(y_data_optimal)
    y_data_worst = Results.worst_reward_sum.squeeze().copy()
    # maximum_gap = y_data_optimal - y_data_worst
    # y_data_optimal = maximum_gap.copy()
    # y_data_optimal /= maximum_gap
    # y_data_worst /= maximum_gap
    plt.plot(x, y_data_optimal[idx], label='Optimal', linewidth='3', color='red')
    plt.plot(x, y_data_worst[idx], label='Worst valid', linewidth='3', color='black')
    # Agent data
    c_list = ['orange', 'gold', 'green', 'blue', 'magenta', 'saddlebrown']
    for i in range(len(agent_list)):
        label_i = 'Agent_' + str(i+1)
        y_data = Results.reward_sum[:, i, :].squeeze().copy()
        # y_data -= y_data_worst
        # y_data /= maximum_gap
        plt.plot(x, y_data[idx], label=label_i, color=c_list[i])

    plt.legend(loc=(1.01, 0.))
    plt.xlabel('Trials', fontdict={'size': '15'})
    plt.ylabel('Performance [m]', fontdict={'size': '15'})
    plt.xlim([1, num_test])  # X축의 범위: [xmin, xmax]
    plt.ylim([0, 400])  # Y축의 범위: [ymin, ymax]
    plt.grid(True)
    plt.show()

    # y_data_optimal = np.full(6, Results.optimal_reward_sum.squeeze().copy().mean())
    # y_data_worst   = np.full(6, Results.worst_reward_sum.squeeze().copy().mean())
    # y_data_agents = np.zeros(6)
    # for i in range(len(agent_list)):
    #     y_data_agents[i] = Results.reward_sum[:, i, :].squeeze().copy().mean()
    #
    # x = [1,2,3,4,5,6]
    # plt.plot(x, y_data_optimal, color='red', linewidth='3.5', label='Optimal ')
    # plt.plot(x, y_data_worst, color='black', linewidth='3.5', label='Worst valid case')
    # plt.plot(x, y_data_agents, color='navy', linewidth='0', marker='o', label='Agents')
    # plt.xlabel('Agent Number')
    # plt.ylabel('Average Performance [m]')
    # plt.legend(loc=(0.75, 0.1))
    # plt.grid(True)
    # plt.show()

    print('The End')
