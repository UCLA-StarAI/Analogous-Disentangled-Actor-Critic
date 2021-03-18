import random
# import roboschool
import gym
import gym.spaces as spaces
import numpy as np
from collections import deque
import os
import time

from envs.AtariPreprocessor import PreprocessAtariStates
# from envs.tapLogicEnv.tapLogicEnv import tapLogicEnv
from envs.RewardShaper import RewardShaper

from utils.LevelFileExporter import LevelFileExporter

from envs.myEnvs.MyEnvs import MyEnvs


class EnvironmentWrapper():
    def __init__(self, env_name, seed = 123, extra_info = {}):
        self.extra_info = extra_info
        self.env_name = env_name
        if len(env_name) >= 16 and env_name[:16] == "HappyElimination":
            if len(env_name) > 16:
                extra_info["extra_info"]["env_idx"] = int(env_name[17:])

            self.env = tapLogicEnv(
                level_file_name = extra_info["level_file_name"],
                state_mode = extra_info["state_mode"],
                action_mode = extra_info["action_mode"],
                reward_mode = extra_info["reward_mode"],
                terminal_mode = extra_info["terminal_mode"],
                extra_info = extra_info["extra_info"]
            )

            if extra_info["train_multiple_levels"]:
                self.train_multiple_levels = True
                self.multiple_level_range = extra_info["multiple_level_range"]
            else:
                self.train_multiple_levels = False

            self.env_type = "HappyElimination"

            self.level_file_exporter = LevelFileExporter(extra_info["level_index"])
            self.enable_record = False
        elif env_name == "CartPole-v0":
            self.env = gym.make("CartPole-v0")
            self.env_type = "gym"
            self.env.seed(seed)

            self.gym_max_episode_steps = self.env._max_episode_steps
        else:
            try:
                # Gym environment
                self.env = gym.make(env_name)
                self.env_type = "gym"
                self.env.seed(seed)

                self.gym_max_episode_steps = self.env._max_episode_steps
            except gym.error.Error:
                if MyEnvs.compatible(env_name):
                    self.env = MyEnvs(env_name)
                    self.env_type = "myenv"
                    self.env.seed(seed)
                else:
                    raise NotImplementedError

        if self.env_type == "HappyElimination":
            # State space
            self.observation_space = {
                "Img": self.env.state_shape,
                "Vec": self.env.vecState_shape
            }

            # Action space
            self.action_mode = "Discrete"
            self.action_n = self.env.action_num
        elif env_name == "CartPole-v0":
            self.observation_space = self.env.observation_space.shape
            self.action_space = gym.make("Pendulum-v0").action_space
            self.action_space.low = -1.0
            self.action_space.high = 1.0

            self.action_mode = "Continuous"
            self.action_dim = self.action_space.shape[0]
            self.action_range = [self.action_space.low, self.action_space.high]
        elif self.env_type == "gym":
            # State space
            if not isinstance(self.env.observation_space, spaces.Box):
                raise RuntimeError("Unknown state space {}".foramt(type(self.env.observation_space)))

            self.observation_space = self.env.observation_space.shape
            if len(self.observation_space) == 3:
                self.observation_space = (4, 84, 84)

                self.last_atari_frames = deque(maxlen = 3)
                for _ in range(3):
                    self.last_atari_frames.append(np.zeros([1, 84, 84]))

            # Action space
            if isinstance(self.env.action_space, spaces.Discrete):
                self.action_mode = "Discrete"
                self.action_n = self.env.action_space.n
            elif isinstance(self.env.action_space, spaces.Box):
                self.action_mode = "Continuous"
                self.action_dim = self.env.action_space.shape[0]
                self.action_range = [self.env.action_space.low[0], self.env.action_space.high[0]]
            else:
                raise RuntimeError("Unknown action space {}".format(type(self.env.action_space)))
        elif self.env_type == "myenv":
            self.observation_space = self.env.observation_space
            self.action_mode = self.env.action_mode
            if self.action_mode == "Continuous":
                self.action_dim = self.env.action_dim
                self.action_range = self.env.action_range
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        self.rewardShaper = RewardShaper(env_name)

        # Start from checkpoint countdown
        self.start_from_checkpoint_count = 0
        self.cool_down = 0

        self.progress_condition = None
        self.scheduled_count = 0
        self.scheduled_cooldown = 0
        self.start_listening = False

    def reset(self, info = dict()):
        if self.env_type == "HappyElimination":
            if self.start_listening:
                self.start_listening = False
                self.progress_condition = None
                self.scheduled_count = 0
                self.scheduled_cooldown = 0

                self.start_from_checkpoint_count = 0
                self.cool_down = 0
            else:
                self.start_listening = True

            if self.train_multiple_levels:
                if "level_idx" not in info:
                    n = np.random.randint(self.multiple_level_range[0], self.multiple_level_range[1] + 2)
                else:
                    n = info["level_idx"]

                if self.start_from_checkpoint_count > 0:
                    state = self.load_check_point()
                    if state is None:
                        state = self.env.reset("envs/tapLogicEnv/levels/" + str(n) + ".txt")
                    else:
                        self.start_from_checkpoint_count -= 1
                else:
                    state = self.env.reset("envs/tapLogicEnv/levels/" + str(n) + ".txt")
                    if self.cool_down > 0:
                        self.cool_down -= 1
            else:
                if self.start_from_checkpoint_count > 0:
                    state = self.load_check_point()
                    if state is None:
                        state = self.env.reset()
                    else:
                        self.start_from_checkpoint_count -= 1
                else:
                    state = self.env.reset()
                    if self.cool_down > 0:
                        self.cool_down -= 1

            if self.enable_record:
                self.level_file_exporter.reset_record(self.env.viewParser)
        # elif self.env_name == "CartPole-v0":

        elif self.env_type == "gym":
            state = self.env.reset()

            if len(self.observation_space) == 3:
                state = PreprocessAtariStates(state)

                for _ in range(3):
                    self.last_atari_frames.append(np.zeros([1, 84, 84]))

                origin_state = state
                state = np.concatenate(
                    (state, self.last_atari_frames[0], self.last_atari_frames[1], self.last_atari_frames[2]),
                    axis = 0
                )

                self.last_atari_frames.append(origin_state)
        elif self.env_type == "myenv":
            state = self.env.reset()
        else:
            raise NotImplementedError()

        return state

    def step(self, action):
        if self.env_type == "HappyElimination":
            if self.start_listening and self.env.check_progress_condition(self.progress_condition):
                flag = self.check_point()

                if not flag:
                    self.start_from_checkpoint_count = 0
                    self.cool_down = 0
                else:
                    self.start_from_checkpoint_count = self.scheduled_count
                    self.cool_down = self.scheduled_cooldown

                self.start_listening = False
                self.progress_condition = None
                self.scheduled_count = 0
                self.scheduled_cooldown = 0

            if self.extra_info["state_mode"] == 0:
                if isinstance(action, int):
                    action = [action // self.env.boardSize[1], action % self.env.boardSize[1]]
                elif len(action) == 1:
                    action = [action[0] // self.env.boardSize[1], action[0] % self.env.boardSize[1]]

            next_state, reward, done, info = self.env.step(action)

            if self.enable_record and not info["unchanged"]:
                self.level_file_exporter.record_next(self.env.viewParser, info["action_for_viewer"])
        elif self.env_name == "CartPole-v0":
            old_action = action
            if action > 0.5:
                action = 1
            elif action < -0.5:
                action = 0
            else:
                action = random.randint(0, 1)

            next_state, reward, done, info = self.env.step(action)

            reward = -1.0 if done else 0.1
            reward -= 0.1 * old_action[0] + 0.05 * old_action[0] ** 2
        elif self.env_type == "gym":
            
            next_state, reward, done, info = self.env.step(action)

            if len(self.observation_space) == 3:
                next_state = PreprocessAtariStates(next_state)

                origin_next_state = next_state
                next_state = np.concatenate(
                    (next_state, self.last_atari_frames[0], self.last_atari_frames[1], self.last_atari_frames[2]),
                    axis = 0
                )

                self.last_atari_frames.append(origin_next_state)
        elif self.env_type == "myenv":
            next_state, reward, done, info = self.env.step(action)
        else:
            raise NotImplementedError()

        return next_state, reward, done, info

    def enable_recording(self):
        self.enable_record = True

    def save_record(self):
        self.level_file_exporter.store_file()

    def render(self, render_double = False, agent_hock = None):
        if self.env_type == "HappyElimination":
            self.env.render(render_double = render_double, agent_hock = agent_hock)
        elif self.env_type == "gym":
            self.env.render(mode = "human")
        elif self.env_type == "myenv":
            pass
        else:
            raise NotImplementedError()

    def get_action_type(self):
        return self.action_mode

    def check_point(self, id):
        if self.env_type == "HappyElimination":
            return self.env.check_point(id)
        else:
            raise NotImplementedError("Env does not support check_point")

    def load_check_point(self, id):
        if self.env_type == "HappyElimination":
            return self.env.load_check_point(id)
        else:
            raise NotImplementedError("Env does not support check_point")

    def clear_check_points(self):
        if self.env_type == "HappyElimination":
            return self.env.clear_check_point()
        else:
            raise NotImplementedError("Env does not support clear_check_point")

    def check_point_to_string(self):
        if self.env_type == "HappyElimination":
            return self.env.check_point_to_string()
        else:
            raise NotImplementedError("Env does not support check_point")

    def load_check_point_from_string(self, raw_emulator, raw_viewParser):
        if self.env_type == "HappyElimination":
            return self.env.load_check_point_from_string(raw_emulator, raw_viewParser)
        else:
            raise NotImplementedError("Env does not support check_point")

    def enable_concentration_learning(self, progress, count, cool_down):
        if self.cool_down > 0:
            return False

        if self.env.need_concentration_learning():
            self.progress_condition = progress
            self.scheduled_count = count
            self.scheduled_cooldown = cool_down

    def get_version_and_AI_from_file_name(self, filename):
        folders = os.path.split(filename)
        assert folders[0] == "envs"
        assert folders[1] == "tapLogicEnv"
