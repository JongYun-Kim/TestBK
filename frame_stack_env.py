"""
Frame_Stacked Environments
    1. CartPole-v1 FO FrameStacked
    2. CartPole-v1 PO FrameStacked
    3. Pendulum-v1 FO FrameStacked
    4. Pendulum-v1 PO FrameStacked
"""
import numpy as np
from gym.spaces import Box

from gym.envs.classic_control import CartPoleEnv, PendulumEnv
# from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
# from ray.rllib.examples.env.stateless_pendulum import StatelessPendulum


class CartPoleFrameStacked(CartPoleEnv):
    """
    Frame stacked CartPole environment
    """

    def __init__(self,
                 num_stack:int,
                 is_fully_observable:bool=True
                 ):
        super().__init__()

        if is_fully_observable:
            high = np.array(
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians * 2,
                    np.finfo(np.float32).max,
                ],
                dtype=np.float32,
            )
        else:
            high = np.array(
                [
                    self.x_threshold * 2,
                    self.theta_threshold_radians * 2,
                ],
                dtype=np.float32,
            )
        high = np.tile(high, num_stack)
        '''
        obs = np.array(
                       [o_t, o_t-1, ..., o_t-num_stack+1]
                       )
        '''
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        self.num_stack = num_stack
        self.unit_obs_size = 4 if is_fully_observable else 2
        self.obs_stacked = None  # (num_stack x obs_unit_size) 행렬
        self.is_fully_observable = is_fully_observable

    def reset(self):
        # Get initial state by running original reset
        super().reset()
        # Get zeros for obs_stacked
        # BE CAREFUL TO SQUEEZE IT WHEN RESHAPE
        self.obs_stacked = np.zeros((self.num_stack, self.unit_obs_size), dtype=np.float32)
        # Stack the current state at the top
        self._stack_states()

        return np.array(self.obs_stacked.flatten(), dtype=np.float32)  # flatten copies and squeezes it

    def step(self, action):
        # Update self.state
        _, reward, done, info = super().step(action)
        # Get Stacked obs
        self._stack_states()

        return np.array(self.obs_stacked.flatten(), dtype=np.float32), reward, done, info

    def _stack_states(self):
        """
        This pushes the current self.state into the stack
        :return: a copied obs_stacked into a 1-D array
        """
        # Push it into the time dimension
        self.obs_stacked[1:, :] = self.obs_stacked[:-1, :]
        # Put the current state on the top
        self.obs_stacked[0, :] = \
            np.array(self.state if self.is_fully_observable else [self.state[0], self.state[2]],
                     dtype=np.float32
                     )  # np.array() copies the arg


class PendulumFrameStacked(PendulumEnv):

    def __init__(self,
                 num_stack: int,
                 is_fully_observable: bool = True
                 ):
        super().__init__()

        if is_fully_observable:
            high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        else:
            high = np.array([1.0, 1.0], dtype=np.float32)
        high = np.tile(high, num_stack)
        '''
        obs = np.array(
                       [o_t, o_t-1, ..., o_t-num_stack+1]
                       )
        '''
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        self.num_stack = num_stack
        self.unit_obs_size = 3 if is_fully_observable else 2
        self.obs_stacked = None  # (num_stack x obs_unit_size) 행렬
        self.is_fully_observable = is_fully_observable

    def reset(self):
        # Get initial state by running original reset
        super().reset()
        # Get zeros for obs_stacked
        # BE CAREFUL TO SQUEEZE IT WHEN RESHAPE
        self.obs_stacked = np.zeros((self.num_stack, self.unit_obs_size), dtype=np.float32)
        # Stack the current state at the top
        self._stack_states()

        return np.array(self.obs_stacked.flatten(), dtype=np.float32)  # flatten copies and squeezes it

    def step(self, action):
        # Update self.state
        _, reward, done, info = super().step(action)
        # Get Stacked obs
        self._stack_states()

        return np.array(self.obs_stacked.flatten(), dtype=np.float32), reward+6.0, done, info

    def _stack_states(self):
        """
        This pushes the current self.state into the stack
        :return: a copied obs_stacked into a 1-D array
        """
        obs_from_state = self._get_obs()
        # Push it into the time dimension
        self.obs_stacked[1:, :] = self.obs_stacked[:-1, :]
        # Put the current state on the top
        self.obs_stacked[0, :] = \
            np.array(obs_from_state if self.is_fully_observable else [obs_from_state[0], obs_from_state[1]],
                     dtype=np.float32
                     )  # np.array() copies the arg
