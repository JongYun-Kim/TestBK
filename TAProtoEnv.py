"""
Env 설명: 정의된 클래스를 보세요.
"""
# import gym
from gym import Env
from gym.spaces import MultiDiscrete
import numpy as np
# import random
# import matplotlib.pyplot as plt
from collections import defaultdict


class TAProtoEnv(Env):
    """
    - Gym Env에 상속된 클래스 (step, render, reset)
    - 시나리오: 정해진 수의 robot들이 랜덤한 초기점에서 시작해서 공간에 랜덤으로 산포된 task들의 위치를 모두 방문하는 시나리오.

    Note (1): RL 관점에서는 주어진 robot들을 central한 하나의 single RL agent가 environment와 interaction함.

    Note(2): robot이 움직이고 task가 존재하는 positions은 모두 discrete하며 100*100 grid로 구성됨.

    - RL formulation:
        1) State: [(positions of the tasks),(positions of the robots),(task_done flags)]
        2) Action: [(next target task numbers of the robots)]
        3) Reward: Sum of distance-based reward of each robot
    """

    def __init__(self, num_uav=2, num_task=3, is_stationary=True, ns_type='stationary'):
        """
        :param num_uav: robot의 숫자
        :param num_task: task의 숫자
        """
        # Define the action space.
        next_task_4_robot_space = np.full((1, num_uav), num_task+1)  # Action upperbounds; 각 로봇이 테스크 넘버를 decision 으로 함. 0일 경우 아무것도 안함.
        self.action_space = MultiDiscrete(np.array(next_task_4_robot_space[0]))

        # Define the state(observation) space.
        # Concatenating them, it consists of three vectors as below.
        # State = [(positions of the tasks),(positions of the robots),(task_completion)]
        #
        # The three vectors are defined
        task_pos_space = np.full((1, 2 * num_task), 100)  # bound of the first vector
        robot_pos_space = np.full((1, 2 * num_uav), 100)  # bound of the second vector
        done_task_space = 2 * np.ones((1, num_task))  # bound of the third vector: binary
        # Concatenate them into a single vector, the action space.
        action_space_bound = np.concatenate((task_pos_space[0], robot_pos_space[0], done_task_space[0]))
        self.observation_space = MultiDiscrete(np.array(action_space_bound))

        # Initialize the variables
        self.num_uav = num_uav
        self.num_task = num_task
        self.state = []
        self.task_pos = []  # task positions
        self.task_pos_x = []  # x-coord of the task positions
        self.task_pos_y = []  # y-coord of the task positions
        self.robot_pos = []  # robot positions
        self.robot_pos_x = []  # x coord of the robot positions
        self.robot_pos_y = []  # y coord of the robot positions
        self.done_task = []  # task completions
        self.travel_dist = np.zeros(num_uav)  # 로봇 별로 움직인 거리 1d array
        self.time_step = 0

        # Define stationarity of the environment
        self.is_stationary = is_stationary
        self.ns_type = ns_type

    def step(self, action):
        self.time_step += 1
        if self.is_stationary:
            reward, done, info = self.step_stationary(action)
        else:
            reward, done, info = self.step_non_stationary(action)

        # Return step information as a gym env
        return self.state, reward, done, info

    def step_stationary(self, action):
        # Apply the action to the environment
        # State update
        for agent, target_task in enumerate(action):
            if target_task == 0:
                self.travel_dist[agent] = 0
            else:
                x_pri = self.robot_pos_x[agent]


        # State 업데이트 (task completions 제외)
        # Compute travel distance of each robot given the action
        for i in range(self.num_uav):  # 각 로봇에 대해서.. (robot_i)
            robot_decision = action[i]
            if robot_decision == 0:  # robot_i 가 아무것도 안하기로 했다면.. (위치 고수)
                self.travel_dist[i] = 0
                # continue
            else:  # robot_i가 특정한 결정을 했다면..
                x_pri = self.robot_pos_x[i]
                y_pri = self.robot_pos_y[i]
                # Update the robot position
                self.robot_pos_x[i] = self.task_pos_x[robot_decision - 1]
                self.robot_pos_y[i] = self.task_pos_y[robot_decision - 1]
                x_post = self.robot_pos_x[i]
                y_post = self.robot_pos_y[i]
                # Compute the travel distance of robot_i
                self.travel_dist[i] = np.sqrt((x_pri - x_post) ** 2 + (y_pri - y_post) ** 2)

        # Compute reward based on the travel distances
        # TODO: Reward normarlization 필요 할지도?..
        rewards = np.zeros(self.num_uav)
        for i in range(self.num_uav):  # robot_i에 대해서..
            if action[i] == 0:  # 움직이지 않기로 한 경우
                rewards[i] = 0  # 움직이지 않기로 했다면 보상을 하지 않음
                continue
            if self.done_task[action[i] - 1] == 1:  # 움직이긴 했는데 다음 목적지가 이미 끝난 task 라면..
                rewards[i] += - self.travel_dist[i]  # 헛수고한 만큼 패널티
                continue
            rewards[i] += 142 - self.travel_dist[i]  # 최대 이동거리 쯤 되는 142보다 덜가서 도착하여 save한 거리만큼 보상

        # See the link below for the function and the next FOR loop, not the one in the function
        # https://stackoverflow.com/a/5419576
        def list_duplicates(seq):
            tally = defaultdict(list)
            for i, item in enumerate(seq):
                tally[item].append(i)
            return ((key, locs) for key, locs in tally.items()
                    if len(locs) > 1)

        for dup in sorted(list_duplicates(action)):
            if dup[0] == 0:  # 만약 선택한 task가 0으로 중복된거라면.. 그냥 계산 스킵
                continue
            # duplicated_rewards = rewards[dup[1]]
            # reward = reward - sum(duplicated_rewards) + max(duplicated_rewards)
            rewards[dup[1]] = -self.travel_dist[dup[1]]  # 먄약 선택한 task가 다른 놈이랑 겹치면 음의 리워드
        reward = sum(rewards)

        # task_completion 업데이트 해서 state 마저 업데이트 하기
        # Reward 계산에서 done task 업데이트 하기 전의 값이 필요해서 지금 늦게서야 함.
        for i in action:
            if i == 0:
                continue
            else:
                self.done_task[i - 1] = 1

        # Check if shower is done
        if self.time_step >= self.num_task or np.sum(self.done_task) == self.num_task:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information as a gym env
        return reward, done, info

    def step_non_stationary(self, action):
        """
        Update the environment with a given action in the presence of NS.

        - Type(1): moving targets (dynamics: )

        - Type(2): loss or gain of agents and targets
        - Type(3): exogenous changes: wind (
        - Type(4): changes in the reward function
        """
        # State update (1): task position update

        # State update (2): robot position update

        # State update (3): task completion update

        # Reward update (1): travel distance-based

        # Reward update: (2): penalties

        # Done update
        if self.time_step >= self.num_task or np.sum(self.done_task) == self.num_task:
            done = True
        else:
            done = False
        # Info update
        info = {}
        return reward, done, info

    def render(self, mode="human"):
        # Implement Viz
        pass
        # Agent plot
        # Task plot

    def reset(self):
        # Initialize state
        self.state = np.array(self.observation_space.sample())
        self.task_pos = self.state[0:2*self.num_task]
        self.task_pos_x = self.task_pos[0:self.num_task]
        self.task_pos_y = self.task_pos[self.num_task:2*self.num_task]
        self.robot_pos = self.state[2*self.num_task:2*(self.num_task+self.num_uav)]
        self.robot_pos_x = self.robot_pos[0:self.num_uav]
        self.robot_pos_y = self.robot_pos[self.num_uav:2*self.num_uav]
        self.done_task = self.state[2*(self.num_task+self.num_uav):2*(self.num_task+self.num_uav)+self.num_task]

        # task 완료를 모두 0으로 만들어줌
        for i in range(self.num_task):
            self.state[2*(self.num_uav+self.num_task)+i] = 0
        self.time_step = 0

        return self.state

    def bs_todo_list(self):
        # TODO (1): Create render func to view the results
        # TODO (2): Switch the action space from MultiDiscrete to (Single)Discrete space
        # TODO (3):
        pass