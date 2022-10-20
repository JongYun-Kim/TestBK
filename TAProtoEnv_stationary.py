"""
Env 설명: 정의된 클래스를 보세요.
"""
# import gym
from gym import Env
from gym.spaces import MultiDiscrete
import numpy as np
# import copy
# import random
# import matplotlib.pyplot as plt
from collections import defaultdict


class TAProtoEnvStationary(Env):
    """
    - Gym Env에 상속된 클래스 (step, render, reset)
    - 시나리오: 정해진 수의 robot들이 랜덤한 초기점에서 시작해서 공간에 랜덤으로 산포된 task들의 위치를 모두 방문하는 시나리오.

    Note(1): RL 관점에서는 주어진 robot들을 central한 하나의 single RL agent가 environment와 interaction함.

    Note(2): robot이 움직이고 task가 존재하는 positions은 모두 discrete하며 100*100 grid로 구성됨.

    - RL formulation:
        1) State: [(positions of the tasks),(positions of the robots),(task_done flags)]
        2) Action: [(next target task numbers of the robots)]
        3) Reward: Sum of distance-based reward of each robot
    """

    def __init__(self, num_uav=2, num_task=3):
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
        self.initial_state = []
        self.task_pos = []  # task positions
        self.task_pos_x = []  # x-coord of the task positions
        self.task_pos_y = []  # y-coord of the task positions
        self.robot_pos = []  # robot positions
        self.robot_pos_x = []  # x coord of the robot positions
        self.robot_pos_y = []  # y coord of the robot positions
        self.done_task = []  # task completions
        self.travel_dist = np.zeros(num_uav)  # 로봇 별로 움직인 거리 1d array
        self.time_step = 0
        self.max_travel_dist = 142

    def step(self, action):
        self.time_step += 1
        # State update (w/o task completions )
        # Compute travel distance of each robot given the action
        for i in range(self.num_uav):  # for each uav_i..
            robot_decision = action[i]
            if robot_decision == 0:  # robot_i 가 아무것도 안하기로 했다면.. (위치 고수)
                self.travel_dist[i] = 0
                # continue
            else:  # robot_i가 특정한 결정을 했다면..
                x_pri = self.robot_pos_x[i]
                y_pri = self.robot_pos_y[i]
                # print(f"IN_STEP_(1): previous position of UAV_{i+1} = ({x_pri},{y_pri})")
                # Update the robot position
                self.robot_pos_x[i] = self.task_pos_x[robot_decision-1]
                self.robot_pos_y[i] = self.task_pos_y[robot_decision-1]
                x_post = self.robot_pos_x[i]
                y_post = self.robot_pos_y[i]
                # Compute the travel distance of robot_i
                # print(f"IN_STEP_(2): previous position of UAV_{i+1} = ({x_pri},{y_pri})")
                # print(f"IN_STEP_(3): current  position of UAV_{i+1} = ({x_post},{y_post})")
                self.travel_dist[i] = np.sqrt((x_pri-x_post)**2 + (y_pri-y_post)**2)
                # print(f"IN_STEP_(4): travel distance of UAV_{i+1} = {(self.travel_dist[i]):>0.2f}")
                # print()

        # Compute reward based on the travel distances
        # TODO: Reward normalization 필요 할지도?..
        rewards = np.zeros(self.num_uav)
        for i in range(self.num_uav):  # robot_i에 대해서..
            if action[i] == 0:  # 움직이지 않기로 한 경우
                rewards[i] = 0  # 움직이지 않기로 했다면 보상을 하지 않음
                continue
            if self.done_task[action[i]-1] == 1:  # 움직이긴 했는데 다음 목적지가 이미 끝난 task 라면..
                rewards[i] += - self.travel_dist[i]  # 헛수고한 만큼 패널티
                continue
            rewards[i] += self.max_travel_dist - self.travel_dist[i]  # 최대 이동거리 쯤 되는 142보다 덜가서 도착하여 save한 거리만큼 보상

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
                self.done_task[i-1] = 1

        # Check if episode is done
        if self.time_step >= self.num_task:
            done = True
            # 아래의 패널티 줄꺼면 if 조건에서 등호를 생략해야함..
            # reward = (self.num_task - np.sum(self.done_task)) * (-self.max_travel_dist*0.5)
        elif np.sum(self.done_task) == self.num_task:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information as a gym env
        return np.copy(self.state), reward, done, info

    def render(self, mode="human"):
        # Implement Viz
        pass
        # Agent plot
        # Task plot

    def reset(self, is_force_reset=False, target_state=np.array([])):
        if is_force_reset:
            if np.any(target_state):  # if the target_state is given
                # TODO: size check has to be done..
                self.state = target_state
                print(" IN_RESET: Make sure to set up your <TIME_STEP> and <INITIAL_STATE>!!")
            else:  # if the target_state is not given; empty
                # Just get it initialized
                if np.any(self.initial_state): # has been initialized before
                    self.state = np.copy(self.initial_state)
                    self.time_step = 0
                else:  # never been initialized but forced to reset
                    # then initialize it randomly anyway
                    print(" IN_RESET: Forced to reset without target state but never been reset before")
                    self.state = np.array(self.observation_space.sample())
                    self.initial_state = np.copy(self.state)
        else:
            # Initialize your state
            self.state = np.array(self.observation_space.sample())

        # Rearrange other elements of the state
        self.task_pos = self.state[0:2*self.num_task]
        self.task_pos_x = self.task_pos[0:self.num_task]
        self.task_pos_y = self.task_pos[self.num_task:2*self.num_task]
        self.robot_pos = self.state[2*self.num_task:2*(self.num_task+self.num_uav)]
        self.robot_pos_x = self.robot_pos[0:self.num_uav]
        self.robot_pos_y = self.robot_pos[self.num_uav:2*self.num_uav]
        self.done_task = self.state[2*(self.num_task+self.num_uav):2*(self.num_task+self.num_uav)+self.num_task]

        if not is_force_reset:
            # task 완료를 모두 0으로 만들어줌
            for i in range(self.num_task):
                self.state[2*(self.num_uav+self.num_task)+i] = 0

        if not is_force_reset:
            # Reset the time step
            self.time_step = 0
            self.initial_state = np.copy(self.state)
        else:
            if not self.time_step==0:
                print(" THE ENVIRONMENT HAS BEEN FORCE-RESET WITHOUT A TIME-STEP RESET !!!!!!!\n")

        return np.copy(self.state)

    def get_optimum(self):
        # TODO: Write a code to get the optimal solution of a given problem for visualizing gaps
        # TODO: This does not get the optimal solution yet..
        if not (self.num_uav == 2 and self.num_task == 3):
            return False

        a00 = np.array([0, 0])
        a01 = np.array([0, 1])
        a02 = np.array([0, 2])
        a03 = np.array([0, 3])
        a10 = np.array([1, 0])
        a12 = np.array([1, 2])
        a13 = np.array([1, 3])
        a20 = np.array([2, 0])
        a21 = np.array([2, 1])
        a23 = np.array([2, 3])
        a30 = np.array([3, 0])
        a31 = np.array([3, 1])
        a32 = np.array([3, 2])

        action_set = np.zeros([24, 3, 2])  # action

        action_set[0, :, :]  = [a10, a20, a30]
        action_set[1, :, :]  = [a10, a30, a20]
        action_set[2, :, :]  = [a20, a10, a30]
        action_set[3, :, :]  = [a20, a30, a10]
        action_set[4, :, :]  = [a30, a10, a20]
        action_set[5, :, :]  = [a30, a20, a10]
        action_set[6, :, :]  = [a13, a20, a00]
        action_set[7, :, :]  = [a23, a10, a00]
        action_set[8, :, :]  = [a12, a30, a00]
        action_set[9, :, :]  = [a32, a10, a00]
        action_set[10, :, :] = [a21, a30, a00]
        action_set[11, :, :] = [a31, a20, a00]
        action_set[12, :, :] = [a12, a03, a00]
        action_set[13, :, :] = [a13, a02, a00]
        action_set[14, :, :] = [a21, a03, a00]
        action_set[15, :, :] = [a23, a01, a00]
        action_set[16, :, :] = [a31, a02, a00]
        action_set[17, :, :] = [a32, a01, a00]
        action_set[18, :, :] = [a01, a02, a03]
        action_set[19, :, :] = [a01, a03, a02]
        action_set[20, :, :] = [a02, a01, a03]
        action_set[21, :, :] = [a02, a03, a01]
        action_set[22, :, :] = [a03, a01, a02]
        action_set[23, :, :] = [a03, a02, a01]

        action_set = action_set.astype(int)

        reward_sum_set = np.zeros([24])
        best_reward_sum = -1000
        best_episode = -100

        for episode in range(24):
            # Go back to the initial condition
            # self.time_step = 0
            self.reset(is_force_reset=True)

            # Run the episode
            done = False
            step = 0
            reward_sum = 0
            while not done:
                # Get an action when the observation given
                action = action_set[episode, step, :]
                # Step the env
                _, reward, done, _ = self.step(action)
                reward_sum += reward
                step += 1

            # Check if it's the best one yet
            if reward_sum > best_reward_sum:
                best_reward_sum = reward_sum
                best_episode = episode

            # Save the reward sum
            reward_sum_set[episode] = reward_sum

        # Get the optimal solution
        optimal_episode_actions = action_set[best_episode, :, :]
        optimal_episode_reward = best_reward_sum

        return optimal_episode_actions, optimal_episode_reward, reward_sum_set

    def get_greedy_result(self):
        def get_greedy_action():
            # Get the positions of the UAVs and tasks and task completion flags
            p_uav = np.array(self.robot_pos).reshape(2, -1).T.copy()  # (2x-) 의 uav 위치 행렬
            dones = np.array(self.done_task)  # 모양은 횡단 행렬
            p_task = np.array(self.task_pos).reshape(2, -1).T.copy()  # (2 x -)의 task 위치 행렬
            p_task_remain = p_task[dones == 0]  # completed tasks only!
            task_number_remain = np.arange(self.num_task)
            task_number_remain = task_number_remain[dones == 0]  # 남아있는 task의 task 번호 (0부터시작)

            # Get all the distances between UAVs and tasks
            p_uav_compute = np.repeat(p_uav, repeats=len(p_task_remain), axis=0)
            p_task_remain_compute = np.tile(p_task_remain, (self.num_uav, 1))
            # dist_all: uav1-task1, uav1-task2, ..., uav2-task1, uav2-task2, ..., uav_n-task_m)
            dist_all = np.sqrt(np.sum((p_uav_compute - p_task_remain_compute) ** 2, axis=1))  # 횡 행렬
            idx_best = np.argsort(dist_all)  # 인덱스 정렬 (int64)
            idx_best = idx_best[0:self.num_uav]  # 인덱스 중 필요한 것만 남김
            idx_best_rev = np.flip(idx_best)  # 인덱스 뒤집음; 둘은 연동된 데이터임에 주의하라!!

            # Initialize action
            action_greedy = np.zeros(self.num_uav, int)

            # Get best action from each best idx from dist_all
            for i_rev in idx_best_rev:  # idx_best중 뒤에서 부터 읽어보자
                # dist_all에서의 idx는 남은 task수로 나누면 몫이 uav번호, task_remain번호 (둘다 0 시작)
                (uav, task_remain) = divmod(i_rev, len(p_task_remain))
                # 진짜 task number를 구하여야 함.
                task = task_number_remain[task_remain]
                # Assign the task of the uav into the action array
                action[uav] = task + 1

            return action_greedy

        # Initialize the environment
        self.reset(is_force_reset=True)

        # Run the environment in a loop
        done = False
        step = 0
        greedy_reward_sum = 0
        while not done:
            # Compute action from the observation
            action = get_greedy_action()
            # action 겹치는 경우
            for dup in self.list_duplicates(action):
                # dup[0]: 중복 action, dup[1]: 해당 action의 uav number
                if dup[0] == 0:
                    continue
                for i_dup in dup[1][1:]:
                    action[i_dup] = 0
            # Step the env
            _, reward, done, _ = self.step(action)
            greedy_reward_sum += reward
            step += 1

        return greedy_reward_sum

    def bs_todo_list(self):  # Do not call this method as it does nothing and really is what it is ...
        # TODO (1): Create render func to graphically view the results
        # TODO (2): Switch the action space from MultiDiscrete to (Single)Discrete space to apply other algos
        # TODO (3):
        pass
