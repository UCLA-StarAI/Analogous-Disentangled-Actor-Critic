# import roboschool
import gym
from copy import deepcopy

from envs.myEnvs.CartPoleSwingUp import make


class MyEnvs():
    def __init__(self, env_name):
        self.env_name = env_name

        if env_name == "CartPoleSwingUpContinuous":
            self.env = make("CartPoleSwingUpContinuous")
        elif env_name == "PendulumSparse":
            self.env = gym.make("Pendulum-v0")
        elif env_name == "AcrobotSparse":
            self.env = gym.make("Acrobot-v1")

        self.action_mode = "Continuous"
        self.observation_space = self.env.observation_space.shape
        if env_name == "AcrobotSparse":
            self.action_dim = 1
            self.action_range = [-1.0, 1.0]
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.action_range = [self.env.action_space.low[0], self.env.action_space.high[0]]

    def reset(self):
        return deepcopy(self.env.reset())

    def step(self, action):
        if self.env_name == "AcrobotSparse":
            if action < -0.333:
                action = 0
            elif -0.333 <= action <= 0.333:
                action = 1
            else:
                action = 2

        next_state, reward, done, info = self.env.step(action)
        next_state = deepcopy(next_state)

        if self.env_name == "PendulumSparse":
            reward = 10.0 if next_state[0] > 0.95 else 0.0
            reward -= action[0] * 0.01

        return next_state, reward, done, info

    def seed(self, seed):
        self.env.seed(seed)

    @staticmethod
    def compatible(env_name):
        if env_name == "CartPoleSwingUpContinuous":
            return True
        elif env_name == "PendulumSparse":
            return True
        elif env_name == "AcrobotSparse":
            return True
        else:
            return False
