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


class TAEnvMovingTasksInEpisode(Env):
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

    def __init__(self, num_uav=2, num_task=30, task_vel=8, acceleration_change_point=-1):
        """
        :param num_uav: robot의 숫자
        :param num_task: task의 숫자
        """
        # Define the action space.
        next_task_4_robot_space = np.full((1, num_uav), num_task+1)
        # Action upperbounds; 각 로봇이 테스크 넘버를 decision 으로 함. 0일 경우 아무것도 안함.
        self.action_space = MultiDiscrete(np.array(next_task_4_robot_space[0]))

        # Define the state(observation) space.
        # Concatenating them, it consists of three vectors as below.
        # State = [(positions of the tasks),(positions of the robots),(task_completion)]
        #
        # The three vectors are defined
        task_pos_space = np.full((1, 2 * num_task), 100)  # bound of the first vector [0, 100) = 0-99
        robot_pos_space = np.full((1, 2 * num_uav), 100)  # bound of the second vector
        done_task_space = 2 * np.ones((1, num_task))  # bound of the third vector: binary
        # Concatenate them into a single vector, the action space.
        observation_space_bound = np.concatenate((task_pos_space[0], robot_pos_space[0], done_task_space[0]))
        self.observation_space = MultiDiscrete(np.array(observation_space_bound))

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
        self.task_vel = self.init_task_vel()  # directions of tasks
        self.initial_task_vel = np.copy(self.task_vel)
        self.vel_ratio_task_to_uav = 0.5  # must be below 1 and non-negative (v_t = r * v_u)
        self.move_time = task_vel  # task speeds
        self.speed_has_changed = False
        if acceleration_change_point == -1:
            acceleration_change_point = int(self.num_task /2)
            print("ACP was not declared")
            print(f"ACP = {acceleration_change_point}")
        self.acceleration_change_point = acceleration_change_point
        if self.acceleration_change_point > self.num_task:
            self.acceleration_change_point = int(self.num_task /2)
            print("The ACP param has got wrong!!!\n This must be smaller than num_task")
            print(f"But ACP(in) = {acceleration_change_point}.")
            print(f"This is now: ACP = {self.acceleration_change_point}")

    def in_episode_ns(self):
        # Changes the speed of tasks
        if self.time_step < self.acceleration_change_point:
            # Increase the speed
            self.move_time += 1
        else:
            # Decrease the speed
            # You must be careful with negative values if not intended
            self.move_time -= 1

    def move_tasks(self):
        move_time = self.move_time  # 당장은 task 모두가 같은 일정한 거리를 한 time_step에서 움직인다고 가정
        # Compute new positions of the tasks
        self.task_pos += self.task_vel * move_time
        self.take_into_grid()  # brings the tasks into the boundary of the grid-world

    def take_into_grid(self):
        # This method brings all tasks outside the boundary into the grid of our interest
        # And the velocities as well accordingly
        changed = 1
        has_been_changed = False
        while changed:
            changed = 0
            for i, x in enumerate(self.task_pos):
                if x > 99:
                    self.task_pos[i] = 198 - x
                    self.task_vel[i] = - self.task_vel[i]
                    changed += 1
                    has_been_changed = True
                elif x < 0:
                    self.task_pos[i] = - x
                    self.task_vel[i] = - self.task_vel[i]
                    changed += 1
                    has_been_changed = True

        return has_been_changed  # returns if the task positions have been changed in this call

    def get_distances(self, action):
        # Computes distances between each uav and its assigned task given the joint action
        # Get next target positions
        x_u_next = self.task_pos_x[action-1].copy()
        y_u_next = self.task_pos_y[action-1].copy()
        # Correct the stationary UAVs' target (the UAVs don't move)
        stationary_uav_idx = np.where(action == 0)
        x_u_next[stationary_uav_idx] = self.robot_pos_x[stationary_uav_idx].copy()
        y_u_next[stationary_uav_idx] = self.robot_pos_y[stationary_uav_idx].copy()

        # Get the distances between each uav and its assigned task
        distances = np.sqrt((x_u_next-self.robot_pos_x)**2 + (y_u_next-self.robot_pos_y)**2)

        return distances, x_u_next, y_u_next  # traveled distances of the UAVs in a 1-D array

    def apply_penalty(self, action, rewards_tot):
        # Impose penalty (1): target task duplications
        for dup in self.list_duplicates(action):  # dup[0]: task number, dump[1]: uav number in index
            if dup[0] == 0:  # 만약 선택한 task가 0으로 중복된거라면.. 그냥 계산 스킵
                continue
            # rewards_tot[dup[1]] = - (rewards_stationary[dup[1]] + rewards_non_stationary[dup[1]])
            rewards_tot[dup[1]] -= self.max_travel_dist
            # 먄약 선택한 task가 다른 놈이랑 겹치면 음의 리워드
        # Impose penalty (2): completed task implementations
        for uav_i, target_task in enumerate(action):  # robot_i에 대해서..
            if target_task == 0:  # 움직이지 않기로 한 경우
                rewards_tot[uav_i] -= self.max_travel_dist  # 움직이지 않기로 했다면 패널티
                continue
            if self.done_task[target_task - 1] == 1:  # 움직이긴 했는데 다음 목적지가 이미 끝난 task 라면..
                rewards_tot[uav_i] -= self.max_travel_dist  # 헛수고한 만큼 패널티
                continue

        return rewards_tot

    def step(self, action):
        # Update time step
        self.time_step += 1
        # Initialization of reward array (step reward)
        rewards_tot = np.zeros(self.num_uav)
        rewards_tot += self.max_travel_dist

        # In-episode NS generator: changes the speeds of tasks
        self.in_episode_ns()

        # Apply action:
        # Update UAV positions and get stationary rewards
        # ## rewards_stationary, self.robot_pos_x[:], self.robot_pos_y[:] = self.get_distances(action)
        # self.robot_pos = np.array()
        # ## rewards_tot += -rewards_stationary

        # Update task positions (They are moving)
        self.move_tasks()

        # Update uav positions (move them to each task)
        rewards_non_stationary, self.robot_pos_x[:], self.robot_pos_y[:] = self.get_distances(action)
        rewards_tot += -rewards_non_stationary

        # Apply penalty on the rewards array
        rewards_tot = self.apply_penalty(action, rewards_tot)

        # Get reward scalar from reward of each agent
        rewards = sum(rewards_tot)

        # Update_task completion
        for i in action:
            if i == 0:
                continue
            else:
                self.done_task[i-1] = 1

        # Check if episode is done
        if self.time_step >= self.num_task:
            done = True
        elif np.sum(self.done_task) == self.num_task:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information as a gym env
        return np.copy(self.state), rewards, done, info

        # See https://stackoverflow.com/a/5419576
    def list_duplicates(self, seq):
        tally = defaultdict(list)
        for i, item in enumerate(seq):
            tally[item].append(i)
        return ((key, locs) for key, locs in tally.items()
                if len(locs) > 1)

    def render(self, mode="human"):
        # Implement Viz
        # Agent plot
        # Task plot
        # Other settings
        pass

    def pretty_print_state(self, vel=True, initials=False):
        print(f"robot_pos = \n {self.robot_pos.copy().reshape(2, -1)}")
        print(f"task_pos = \n {self.task_pos.copy().reshape(2, -1)}")
        print(f"task_completion = {self.done_task}")
        if vel:
            print(f"task_vel_init =\n {self.task_vel.copy().reshape(2, -1)}")
        if initials:
            print(f"initial_robot_pos = \n "
                  f"{self.initial_state[2*self.num_task:2*(self.num_task+self.num_uav)].copy().reshape(2,-1)}")
            print(f"initial_task_pos = \n {self.initial_state[0:2*self.num_task].copy().reshape(2,-1)}")
            print(f"initial_task_vel = \n {self.initial_task_vel.copy().reshape(2,-1)}")

    def reset(self, is_force_reset=False, target_state=np.array([]), target_vel=np.array([])):
        if is_force_reset:
            if np.any(target_state):  # if the target_state is given
                # TODO: size check has to be done..
                self.state = target_state
                self.task_vel = target_vel
                print(" IN_RESET: Make sure to set up your <TIME_STEP> and <INITIAL_STATE>!!")
            else:  # if the target_state is not given; empty
                # Just get it initialized
                if np.any(self.initial_state):  # has been initialized before
                    self.state = np.copy(self.initial_state)
                    self.time_step = 0
                    self.task_vel = np.copy(self.initial_task_vel)
                else:  # never been initialized but forced to reset
                    # then initialize it randomly anyway
                    print(" IN_RESET: Forced to reset without target state but never been reset before")
                    self.state = np.array(self.observation_space.sample())
                    self.initial_state = np.copy(self.state)
                    self.task_vel = self.init_task_vel()
                    self.initial_task_vel = np.copy(self.task_vel)
        else:
            # Initialize your state
            self.state = np.array(self.observation_space.sample())
            self.task_vel = self.init_task_vel()
            self.initial_task_vel = np.copy(self.task_vel)

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
            if not self.time_step == 0:
                print(" THE ENVIRONMENT HAS BEEN FORCE-RESET WITHOUT A TIME-STEP RESET !!!!!!!\n")

        return np.copy(self.state)

    def init_task_vel(self):
        # This func randomly generates velocity for each task
        # Generate random vels in a vector
        # target_vel = np.random.randint(1, 2, self.num_task*2)  # magnitude of vel in each axis is either 1 or 2
        # Get signs
        # vel_sign = np.random.randint(0, 2, len(target_vel))*2 - 1
        # Apply the sign
        # target_vel *= vel_sign
        # # Reshape it into 2-dim vectors: (-1, 2)
        # target_vel = np.reshape(target_vel, [-1, 2])
        target_vel = np.random.randint(0, 2, self.num_task*2)*2 - 1  # Get directional vectors with sqrt(2)
        # if the reshape above is commented out, the target_vel is a 1-D numpy array
        return target_vel

    def bs_todo_list(self):  # Do not call this method as it does nothing and really is what it is ...
        # TODO (1): Create render func to graphically view the results
        # TODO (2): Switch the action space from MultiDiscrete to (Single)Discrete space to apply other algos
        # TODO (3):
        pass
