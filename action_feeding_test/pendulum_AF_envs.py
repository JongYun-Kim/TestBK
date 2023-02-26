from gym.spaces import Box
import numpy as np
from ray.rllib.env.env_context import EnvContext
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize


class PendulumActionStackedEnv(PendulumEnv):

    def __init__(self,
                 af: bool,
                 stack_length: int,
                 ):

        # Init the env
        super().__init__()  # It uses g=10.0

        # Fix the observation space if AF required
        if isinstance(af, bool):
            self.af = af
        else:
            raise "[DataTypeError] Please af must be a boolean !!"
        if stack_length < 1:
            raise "stack_length must NOT be smaller than 1"
        if isinstance(stack_length, int) is False:
            print("stack_length has been casted into an int val")
        if self.af:
            self.stack_length = int(stack_length)
            high_org = np.array([1.0, 1.0, 8.0], dtype=np.float32)  # action added at the last (-2.0 - +2.0)
            high_action = 2.0 * np.ones(self.stack_length, dtype=np.float32)
            high = np.append(high_org, high_action)
            self.observation_space = Box(low=-high, high=high, dtype=np.float32)
        else:
            self.stack_length = 0
            # print("stack_length has been overwritten to 0 as af==False.")
        # Get stacked action variable, but it will be generated in reset if required
        self.stacked_actions = None
        self.obs_org = None
        self.obs_tot = None
        self.time_step = None

    def step(self, action):
        # Get obs from the Pend env
        next_obs, reward, done, info = super().step(action)
        # next_obs is [cos(theta), sin(theta), theta-dot]

        # Update the observation
        self.obs_org[:] = next_obs

        # Add the action into the obs
        if self.af:  # when the action feeding is running
            self.stacked_actions[:-1] = self.stacked_actions[1:]
            self.stacked_actions[-1] = np.array(action, dtype=np.float32)
            # obs_stacked = np.append(next_obs, self.stacked_actions)

        self.time_step += 1

        return self.obs_tot, reward, done, info

    def reset(self):
        # Init time step
        self.time_step = 0
        # Get init obs
        init_obs = super().reset()
        # init_obs is [cos(theta), sin(theta), theta-dot (angular velocity)]

        # Define total observation
        self.obs_tot = np.zeros(3 + self.stack_length, dtype=np.float32)
        # Define observation w/o action and the stacked actions
        self.stacked_actions = self.obs_tot[3:]
        self.obs_org = self.obs_tot[:3]
        # Update observation
        self.obs_org[:] = init_obs

        return self.obs_tot
        # return np.copy(self.obs_tot)


class PendulumObsNoise(PendulumActionStackedEnv):

    def __init__(self,
                 config: EnvContext,
                 ):
        """
        :param config: configuration of the environment

        config:
            noise_mode: Uniform, Normal, Clear
                Uniform
                    obs_pos_radius: (float)
                    obs_ang_radius: (float) angular velocity! (i.e. theta_dot)
                Normal
                    obs_pos_noise_std: (float)
                    obs_ang_noise_std: (float)
                Clear
            AF: (bool) Action feeding or not
            stack_length: (int) how many actions are you going to stack

        Template
            config = {AF: True,
                      stack_length: 1,
                      noise_mode: "Uniform",
                      obs_pos_radius: 0.1,
                      obs_pos_radius: 0.1,
                      noise_mode: "Normal",
                      obs_pos_noise_std: 0.05,
                      obs_pos_noise_std: 0.05,
                      noise_mode: "Clear",
                     }
        """
        # Get the config
        self.config_action = config
        # Get mode
        self.noise_mode = config["noise_mode"]
        if self.noise_mode is not ("Uniform" or "Normal" or "Clear"):
            raise "noise_mode must be either Uniform, Normal, or Clear."

        # Init the env
        super().__init__(af=config["AF"], stack_length=config["stack_length"])  # It uses g=10.0 for now

    def _get_new_obs(self, obs_env):
        obs_bound = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # Get new obs
        if self.noise_mode == "Uniform":
            radius_pos = float(self.config_action["obs_pos_radius"])
            radius_ang = float(self.config_action["obs_ang_radius"])
            if min(radius_ang, radius_pos) <= 0:
                raise "obs radius must be positive value"
            obs_radius = np.array([radius_pos, radius_pos, radius_ang], dtype=np.float32)
            obs_corrupted = np.random.uniform(low=obs_env-obs_radius, high=obs_env+obs_radius)
        elif self.noise_mode == "Normal":
            std_pos = self.config_action["obs_pos_noise_std"]
            std_ang = self.config_action["obs_ang_noise_std"]
            std_array = np.array([std_pos, std_pos, std_ang], dtype=np.float32)
            obs_corrupted = np.random.normal(loc=obs_env, scale=std_array)
        else:
            raise "wtf, how?"
        # Clip the new observation
        obs_corrupted = np.clip(a=obs_corrupted, a_min=-obs_bound, a_max=obs_bound, dtype=np.float32)

        return obs_corrupted

    def step(self, action):
        # Step is
        _, reward, done, info = super().step(action)

        # Get the corrupted observation into our observation feeding (the stacked actions excluded)
        if self.noise_mode != "Clear":
            self.obs_org[:] = self._get_new_obs(self.obs_org)

        return self.obs_tot, reward, done, info

    def reset(self):
        pass


class PendulumActionNoise(PendulumActionStackedEnv):

    def __init__(self,
                 config: EnvContext,
                 ):
        """
        :param config: configuration of the environment

        config:
            noise_mode: Uniform, Normal, Clear
                Uniform
                    action_radius: (float)
                Normal
                    action_noise_std: (float)
                Clear
            AF: (bool) Action feeding or not
            stack_length: (int) how many actions are you going to stack

        Template
            config = {AF: True,
                      stack_length: 1,
                      noise_mode: "Uniform",
                      action_radius: 0.15,
                      noise_mode: "Normal",
                      action_noise_std: 0.1,
                      noise_mode: "Clear",
                     }
        """
        # Get the config
        self.config_action = config
        # Get mode
        self.noise_mode = config["noise_mode"]
        if self.noise_mode is not ("Uniform" or "Normal" or "Clear"):
            raise "noise_mode must be either Uniform, Normal, or Clear."

        # Init the env
        super().__init__(af=config["AF"], stack_length=config["stack_length"])  # It uses g=10.0 for now

    def _get_new_action(self, action_obs):
        if self.noise_mode == "Uniform":
            radius = float(self.config_action["action_radius"])
            if radius <= 0:
                raise "action radius must be positive value"
            action_env = np.random.uniform(action_obs-radius, action_obs+radius)
        elif self.noise_mode == "Normal":
            s = self.config_action["action_noise_std"]
            action_env = np.random.normal(loc=action_obs, scale=s)
        elif self.noise_mode == "Clear":
            action_env = action_obs
        else:
            raise "wtf; why?"
        # Clip the action in [-2.0, 2.0]
        action_env = np.clip(action_env, -self.max_torque, self.max_torque, dtype=np.float32)

        return action_env

    def step(self, action):
        # Get the corrupted action
        action_env = self._get_new_action(action_obs=action)
        _, reward, done, info = super().step(action_env)

        # Put the action we know into the obeservation
        if self.af:  # when the action feeding is running
            self.stacked_actions[-1] = np.array(action, dtype=np.float32)

        return self.obs_tot, reward, done, info

    def reset(self):
        pass


class PendulumTransitionNoise(PendulumActionStackedEnv):

    def __init__(self,
                 config: EnvContext,
                 ):
        """
        :param config: configuration of the environment

        config:
            g_mode: (string) Fixed, Uniform, Discrete, Normal
                g_fixed (Fixed): fixed g value
                g_min (Uniform, Discrete): minimum value of g
                g_max (Uniform): maximum value of g
                g_interval (Discrete): interval of the discrete values
                g_max_count (Discrete): how many types of the discrete values you want
                g_mean (Normal): mean of the Gaussian
                g_std (Normal): standard deviation of the Gaussian
            in_episode_ns: (bool) whether g is changing during the episode
            AF: (bool) Action feeding or not
                stack_length: (int) how many actions are you going to stack

        Template
            config = {AF: True,
                      stack_length: 1,
                      g_mode: "Fixed",
                      g_fixed: 9.81,
                      g_mode: "Uniform",
                      g_min: 9.8,
                      g_max: 12.0
                      g_mode: "Discrete",
                      g_min: 9.0,
                      g_interval: 0.1,
                      g_max_count: 21,
                      g_mode: "Normal",
                      g_mean: 9.5,
                      g_std: 1.0,
                     }
        """
        # Set g_bottom
        self.g_bottom = 0.1
        # Get config
        self.config_g = config

        # Init the env
        super().__init__(af=config["AF"], stack_length=config["stack_length"])  # It uses g=10.0 for now

    def _get_new_g(self):
        # Get gravity value
        g_mode = self.config_g["g_mode"]
        if g_mode == "Fixed":
            g = self.config_g["g_fixed"]
        elif g_mode == "Uniform":
            g_min = self.config_g["g_min"]
            g_max = self.config_g["g_max"]
            g = ((g_max-g_min) * np.random.rand()) + g_min
        elif g_mode == "Discrete":
            g_min = self.config_g["g_min"]
            g_interval = self.config_g["g_interval"]
            g_max_count = self.config_g["g_max_count"]
            g = g_min + (np.random.randint(g_max_count)*g_interval)
        elif g_mode == "Normal":
            g = np.random.normal(loc=self.config_g["g_mean"], scale=self.config_g["g_std"])
        else:
            raise "Please, check your g_mode in the env config, man.\n" \
                  "Must be either Fixed, Uniform, Discrete, or Normal"
        # Get the g sample bounded
        if g < self.g_bottom:
            g = self.g_bottom
            print("\n[Gym_Env_Notice] g_bottom has been hit. (0.1)")
            print("[Gym_Env_Notice] Please consider a higher gravity setting.")
            print("[Gym_Env_Notice] Or, it might have reached the bottom "
                  "just because of the stochasticity of your g_mode\n")
        return g

    def step(self, action):
        next_obs, reward, done, info = super().step(action)  # PendulumActionStacked 의 step
        # next_obs is [cos(theta), sin(theta), theta-dot]
        # TODO: In-Episode Non-stationarity if required
        # if self.config_g["in_episode_ns"]:
        #     if self.config_g["g_mode"] is not "Fixed":
        #         self.g = self._get_new_g()
        return next_obs, reward, done, info

    def reset(self):
        # Set gravity value for this environment
        self.g = self._get_new_g()
        # Get init obs stacked
        init_obs = super().reset()
        return init_obs


class PendulumRewardNoise(PendulumActionStackedEnv):

    def __init__(self,
                 config: EnvContext,
                 ):
        """
            :param config: configuration of the environment

            config:
                noise_mode: Coefficient, Discounted, Uniform, Normal, Clear
                    Coefficient: additional params not required for now (use this for evaluation only)
                            angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
                        c1: (float) 1.0 default
                        c2: (float) 0.1 default
                        c3: (float) 0.001 default
                    Discounted:
                        discount_factor: (float)
                    Uniform  #TODO: NOT SUPPORTED YET !!
                        reward_radius: (float)
                    Normal   #TODO: NOT SUPPORTED YET !!
                        reward_noise_std: (float)
                    Clear
                AF: (bool) Action feeding or not
                stack_length: (int) how many actions are you going to stack

            Template
                config = {AF: True,
                          stack_length: 1,
                          noise_mode: "Coefficient",
                          c1: 1.0,
                          c2: 0.1,
                          c3: 0.001,
                          noise_mode: "Discounted",
                          discount_factor: 0.95,
                          noise_mode: "Clear",
                         }
        """
        # Get the config
        self.config_reward = config
        # Get mode
        self.noise_mode = config["noise_mode"]
        if self.noise_mode is not ("Uniform" or "Normal" or "Clear"):
            raise "noise_mode must be either Coefficient, Discounted, Uniform, Normal, or Clear."

        # Init the env
        super().__init__(af=config["AF"], stack_length=config["stack_length"])  # It uses g=10.0 for now

    def _get_new_reward(self, reward_org, action):
        if self.noise_mode == "Coefficient":
            th, thdot = self.state
            c1 = self.config_reward["c1"]
            c2 = self.config_reward["c2"]
            c3 = self.config_reward["c3"]
            u = np.clip(action, -self.max_torque, self.max_torque)[0]
            reward_new = -(c1*(angle_normalize(th) ** 2) + c2 * (thdot ** 2) + c3 * (u ** 2))
        elif self.noise_mode == "Discounted":
            discount_factor = self.config_reward["discount_factor"]
            horizon = 200
            if self.time_step < horizon:
                reward_new = reward_org**(horizon-self.time_step)
            else:
                reward_new = reward_org
        elif self.noise_mode == "Uniform":
            raise "Uniform mode is not supported yet... Please use Coefficient, Discounted, or Clear"
        elif self.noise_mode == "Normal":
            raise "Normal mode is not supported yet..."
        else:
            raise "You are not supposed to be here (_get_new_reward() function). " \
                  "\nThe integrity of this class may have been compromised."
        return reward_new

    def step(self, action):
        # Step
        _, reward, done, info = super().step(action)

        # Get the corrupted observation into our observation feeding (the stacked actions excluded)
        if self.noise_mode != "Clear":
            reward = self._get_new_reward(reward, action)

        return self.obs_tot, reward, done, info

    def reset(self):
        pass


class PendulumPartialObs(PendulumEnv):

    def __init__(self,
                 config: EnvContext,
                 ):
        # config = config or {}
        # g = config.get("g", 10.0)
        self._config_test_val = config["test"]
        if "g" in config:
            g = config["g"]
        else:
            g = 10.0

        super().__init__(g=g)

        # Fix our observation-space (remove angular velocity component).
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        # next_obs == [cos(theta), sin(theta), theta-dot (angular velocity)]
        # next_obs[:-1] == [ cos(theta), sin(theta) ]
        return next_obs[:-1], reward, done, info

    def reset(self):
        init_obs = super().reset()
        # init_obs == [cos(theta), sin(theta), theta-dot (angular velocity)]
        return init_obs[:-1]
