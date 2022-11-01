"""
Env 설명: 정의된 클래스를 보세요.
"""
from gym import Env
from gym.spaces import Box, MultiDiscrete
# from gym.spaces import Box, Discrete
import numpy as np
# import math
# import copy
# import random
# import matplotlib.pyplot as plt
from collections import defaultdict


class TAEnvDynaRand(Env):
    """
    - Gym Env에 상속된 클래스 (step, render, reset)
    - 시나리오: 정해진 수의 UAV-들이 랜덤한 초기점에서 시작해서 공간에 랜덤으로 산포된 task-들의 위치를 모두 방문하는 시나리오.

    Note(1): RL 관점 에서는 주어진 UAV-들을 central-한 하나의 single RL agent-가 environment-와 interaction-함.

    Note(2): UAV 가 움직이고 task-가 존재하는 positions 은 모두 continuous-하며 1*1 grid로 구성됨.

    - RL formulation:
        1) State: [(positions of the tasks),(positions of the robots),(task_done flags)]
        2) Action: [(next target task numbers of the robots)]
        3) Reward: Sum of distance-based reward of each robot
    """
    def __init__(self,
                 num_uav: int,
                 num_task: int,
                 is_moving: bool,
                 task_speed_min: int,
                 task_speed_max: int,
                 is_ns_in_episode: bool,
                 change_point: int,
                 acceleration: int,
                 ):
        """
            :param num_uav: (int) the number of UAVs
            :param num_task: (int) the number of tasks
            :param is_moving: (bool) if the tasks move or not
            :param task_speed: (int) speed of tasks.
            :param is_ns_in_episode: (bool) if in-ep ns or not
            :param change_point: (int) the time step when the acceleration changes
            :param acceleration: (int) acceleration of tasks
        """
        # Define variables constant during the run
        self.num_uav = int(num_uav)
        self.num_task = int(num_task)
        self.penalty = 142  # np.sqrt(2)

        # Define variables changing during the run
        self.task_speed = None
        self.initial_task_direction = None
        self.task_direction = None

        # Define observation space: MultiDiscrete
        task_pos_space = np.full((1, 2 * num_task), 100)  # bound of the first vector [0, 100) = 0-99
        robot_pos_space = np.full((1, 2 * num_uav), 100)  # bound of the second vector
        done_task_space = 2 * np.ones((1, num_task), dtype=int)  # bound of the third vector: binary
        obs_size = np.concatenate((task_pos_space[0], robot_pos_space[0], done_task_space[0]))
        self.observation_space = MultiDiscrete(obs_size)
        # obs_size = 2*self.num_uav + 2*self.num_task + self.num_task
        # self.observation_space = Box(low=np.zeros(obs_size), high=np.ones(obs_size), dtype=float)

        # Define action space
        action_size = np.full(num_uav, num_task+1)
        self.action_space = MultiDiscrete(action_size)
        # self.action_space = Discrete(self.num_uav * (self.num_task+1))

        # Define RL environment variables
        self.state = None
        self.initial_state = None
        self.uav_position = None
        self.task_position = None
        self.task_completion = None
        self.time_step = None
        self.is_moving = is_moving
        self.initial_task_speed = None
        self.task_speed_min = task_speed_min
        self.task_speed_max = task_speed_max
        self.is_ns_in_episode = is_ns_in_episode
        self.change_point = change_point
        self.acceleration = acceleration

    def _in_episode_ns(self):
        """
        This method generates non-stationarity in the environment
        by changing the speed and acceleration of the tasks within an episode
        :return: None
        """
        # Changes the speed of tasks
        if self.time_step < self.change_point:
            # Increase the speed
            self.task_speed += self.acceleration
        else:
            # Decrease the speed
            # You must be careful with negative values if not intended
            self.task_speed -= self.acceleration

    def _move_tasks(self):
        """
        This method moves the tasks by one-step given the task velocity.
        :return: None
        """
        # move the tasks
        self.task_position += self.task_speed * self.task_direction  # +=는 state도 변화시킴. operator에 주의
        self._take_into_grid()

    def _take_into_grid(self):
        """
        This method brings all tasks outside the boundary into the grid of our interest.
        It changes the positions and velocity accordingly.
        :return: (bool) as to whether the task positions have been changed
        """
        changed = 1
        has_been_changed = False
        while changed:
            changed = 0
            for i, x in enumerate(self.task_position.reshape(-1, 1)):
                if x > 99:
                    self.task_position.reshape(-1, 1)[i] = 198 - x  # 1-(x-1)==2-x
                    self.task_direction.reshape(-1, 1)[i] = - self.task_direction.reshape(-1, 1)[i]
                    changed += 1
                    has_been_changed = True
                elif x < 0:
                    self.task_position.reshape(-1, 1)[i] = - x
                    self.task_direction.reshape(-1, 1)[i] = - self.task_direction.reshape(-1, 1)[i]
                    changed += 1
                    has_been_changed = True
        return has_been_changed

    def _move_uavs_to_tasks(self, action):
        """
        This method applies the action to the UAV positions.
        Each UAV moves to the task position according to the given action
        :param action: (np.ndarray) action of the environment
        :return: (np.ndarray) traveled distances of the UAVs, (np.ndarray) updated UAV positions
        """
        # Get next uav positions, which are target task positions
        uav_next_pos = self.task_position[action-1, :].copy()
        # Correct the stationary UAVs' target (the UAVs don't move)
        stationary_uav_index = np.where(action == 0)
        uav_next_pos[stationary_uav_index, :] = self.uav_position[stationary_uav_index, :].copy()

        # Get the distance between each uav and its assigned task
        # traveled distances of the UAVs in a 1-D array
        distances = np.sqrt(np.sum((uav_next_pos-self.uav_position)**2, axis=1))

        return distances, uav_next_pos

    def _apply_penalty(self, action, rewards_tot):
        """
        This methods applies the penalty to the given reward array.
        penalty(1): 다른 UAV와 중복된 task를 고른경우
        penalty(2): 이전 step에서 이미 끝난 task를 고른경우
        Note: penalty(1)과 penalty(2)는 중복으로 가능함
        :param action: (np.ndarray) action array of each UAV; each val means target UAV number (0 is to keep the pos)
        :param rewards_tot: (np.ndarray) reward array of each UAV
        :return: updated reward array
        """
        # Impose penalty (1): target task duplications
        for dup in self._list_duplicates(action):  # dup[0]: task number, dump[1]: uav number in index
            if dup[0] == 0:  # 만약 선택한 task가 0으로 중복된거라면.. 그냥 계산 스킵
                continue
            # rewards_tot[dup[1]] = - (rewards_stationary[dup[1]] + rewards_non_stationary[dup[1]])
            rewards_tot[dup[1]] -= self.penalty
            # 먄약 선택한 task가 다른 놈이랑 겹치면 음의 리워드
        # Impose penalty (2): implementation of completed tasks
        for uav_i, target_task in enumerate(action):  # robot_i에 대해서..
            if target_task == 0:  # 움직이지 않기로 한 경우
                rewards_tot[uav_i] -= self.penalty  # 움직이지 않기로 했다면 패널티
                continue
            if self.task_completion[target_task - 1] == 1:  # 움직이긴 했는데 다음 목적지가 이미 끝난 task 라면..
                rewards_tot[uav_i] -= self.penalty  # 헛수고한 만큼 패널티
                continue
        return rewards_tot

    def _list_duplicates(self, seq):
        """
        This (static) method returns the duplicated elements of the input sequence.
        Please see the link below for more information.
        # https://stackoverflow.com/a/5419576
        :param seq: (list) the array of interest
        :return: (dictionary) key: duplicated element, val: index of the element
        """
        tally = defaultdict(list)
        for i, item in enumerate(seq):
            tally[item].append(i)
        return ((key, locs) for key, locs in tally.items()
                if len(locs) > 1)

    # def step(self, action: np.ndarray):
    def step(self, action: np.ndarray):
        # Update time step
        if self.time_step is None:
            raise Exception("You must reset the environment before step.")
        elif self.time_step < 0:
            raise Exception("Time-step must be a non-negative integer")
        else:
            self.time_step += 1

        # Initialize in-method variables
        rewards_tot = np.zeros(self.num_uav)
        rewards_tot += self.penalty

        # In-episode non-stationarity
        if self.is_ns_in_episode:
            self._in_episode_ns()

        # Move the tasks
        if self.is_moving:
            self._move_tasks()

        # Update uav positions and get traveled distances
        rewards_uavs, self.uav_position[:] = self._move_uavs_to_tasks(action)
        rewards_tot += - rewards_uavs

        # Apply penalty
        rewards_tot = self._apply_penalty(action, rewards_tot)

        # Reward scalar update
        # rewards_tot += rewards_uavs + rewards_penalty
        reward = sum(rewards_tot)

        # Update task completion flags
        for i in action:  # int numpy array
            if i == 0:
                continue
            else:
                self.task_completion[i-1] = 1  # int로 지정했지만 float으로 될꺼긴함

        # Check if the episode is done
        if self.time_step >= self.num_task:
            done = True
        elif np.sum(self.task_completion) == self.num_task:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        return np.copy(self.state), reward, done, info

    def reset(self, do_over=False):
        has_been_reset = False if self.time_step is None else True

        if has_been_reset is not True or do_over is not True:  # 리셋 안해봤거나 반복 안하는 경우
            self.initial_state = self.observation_space.sample()
            self.initial_task_direction = self._init_task_direction()
            self.initial_task_speed = self._init_task_speed()

        self.time_step = 0
        self.state = self.initial_state.copy()
        self.task_speed = self.initial_task_speed
        self.task_direction = self.initial_task_direction.copy()

        # Rearrange the state
        self.uav_position = self.state[0:2*self.num_uav].reshape(-1, 2)
        self.task_position = self.state[2*self.num_uav:2*(self.num_uav+self.num_task)].reshape(-1, 2)
        self.task_completion = self.state[2*(self.num_uav+self.num_task):]

        # Make the task completion flags all 0.0
        # TODO: state-와 init state 모두 에서 task 부분을 0으로 만들-어야함. 튜플-이면 int-지만 나머진 float 0.0
        self.task_completion[:] = 0

        return np.copy(self.state)

    def _init_task_direction(self):
        # theta = np.random.rand(self.num_task, 1) * 2 * np.pi
        # dir_data = np.concatenate([np.cos(theta), np.sin(theta)], 1)
        # dir_vec = np.array(dir_data)
        # return dir_vec
        dir_vec_discrete = (np.random.randint(0, 2, self.num_task*2)*2-1).reshape(-1, 2)
        return dir_vec_discrete

    def _init_task_speed(self):
        initial_speed = np.random.randint(low=self.task_speed_min, high=self.task_speed_max+1)
        return initial_speed

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_greedy_sol(self):
        def get_greedy_action():
            # Get the positions of the UAVs and tasks and task completion flags
            p_uav = self.uav_position.copy()  # (2x-) 의 uav 위치 행렬
            dones = self.task_completion  # 모양은 횡단 행렬
            p_task = self.task_position.copy()  # (2 x -)의 task 위치 행렬
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
                action_greedy[uav] = task + 1

            return action_greedy

        # Initialize the environment
        if self.time_step is None:
            raise Exception("The environment must be reset once, before you get greedy results")
        else:
            self.reset(do_over=True)

        # Run the environment in a loop
        done = False
        step = 0
        greedy_reward_sum = 0
        while not done:
            # Compute action from the observation
            action = get_greedy_action()
            # action 겹치는 경우
            for dup in self._list_duplicates(action):
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

    def get_todo(self):
        # TODO(1):
        # TODO(2):
        print("Please see the TODO note in the method")
        pass
