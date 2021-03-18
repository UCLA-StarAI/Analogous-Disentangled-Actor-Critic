import roboschool
import gym
import gym.spaces as spaces
import numpy as np
from collections import deque
from multiprocessing import Process
from copy import deepcopy
import time

from envs.AtariPreprocessor import PreprocessAtariStates
from envs.tapLogicEnv.tapLogicEnv import tapLogicEnv
from envs.RewardShaper import RewardShaper

from utils.LevelFileExporter import LevelFileExporter


class SingleCoreEnvWrapper(Process):
    def __init__(self, env_name, seed = 123, extra_info = {}, pipe = None):
        super(SingleCoreEnvWrapper, self).__init__()

        extra_extra_info = dict()
        for key, value in extra_info["extra_info"].items():
            extra_extra_info[key] = deepcopy(value)

        env_name = deepcopy(env_name)

        self.extra_info = deepcopy(extra_info)
        self.extra_extra_info = extra_extra_info

        if len(env_name) >= 16 and env_name[:16] == "HappyElimination":
            if len(env_name) > 16:
                extra_info["extra_info"]["env_idx"] = int(env_name[17:])

            self.env = tapLogicEnv(
                level_file_name = extra_info["level_file_name"],
                state_mode = extra_info["state_mode"],
                action_mode = extra_info["action_mode"],
                reward_mode = extra_info["reward_mode"],
                terminal_mode = extra_info["terminal_mode"],
                extra_info = extra_extra_info
            )

            if extra_info["train_multiple_levels"]:
                self.train_multiple_levels = True
                self.multiple_level_range = extra_info["multiple_level_range"]
            else:
                self.train_multiple_levels = False

            self.env_type = "HappyElimination"

            self.level_file_exporter = LevelFileExporter(extra_info["level_index"])
            self.enable_record = False
        else:
            try:
                # Gym environment
                self.env = gym.make(env_name)
                self.env_type = "gym"
                self.env.seed(seed)
            except gym.error.Error:
                raise NotImplementedError()

        if self.env_type == "HappyElimination":
            # State space
            self.observation_space = {
                "Img": self.env.state_shape,
                "Vec": self.env.vecState_shape
            }

            # Action space
            self.action_mode = "Discrete"
            self.action_n = self.env.action_num
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

        self.pipe = pipe

    def run(self):
        while True:
            command, args = self.receive_safe_protocol(self.pipe)

            if command == 0:
                # Reset
                if args is None:
                    item = deepcopy(self.reset())
                    # item = np.zeros([10, 10])
                else:
                    item = deepcopy(self.reset(info = args))
                    # item = np.zeros([10, 10])

                self.send_safe_protocol(self.pipe, 10, item)

            elif command == 1:
                # Step
                next_state, reward, done, info = self.step(args)
                # next_state, reward, done, info = np.zeros([10, 10]), 0, True, dict()
                next_state = deepcopy(next_state)
                reward = deepcopy(reward)
                done = deepcopy(done)
                info = deepcopy(info)
                item = (next_state, reward, done, info)

                self.send_safe_protocol(self.pipe, 11, item)

            elif command == 2:
                # Terminate
                return

            elif command == 3:
                continue

            else:
                raise NotImplementedError()

    def send_safe_protocol(self, pipe, command, args):
        success = False

        while not success:
            pipe.send((command, args))

            ret = pipe.recv()
            if ret == command:
                success = True

    def receive_safe_protocol(self, pipe):
        pipe.poll(None)

        command, args = pipe.recv()
        # print("[slave] received command", command)

        pipe.send(command)
        # print("[slave] send command", command)

        return deepcopy(command), deepcopy(args)

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
        else:
            raise NotImplementedError()

        return state

    def step(self, action):
        if self.env_type == "HappyElimination":
            if self.start_listening and self.env.check_progress_condition(self.progress_condition):
                start = time.clock()
                flag = self.check_point()

                end = time.clock()
                if end - start > 0.1:
                    print("Takes too long for checkpoint")
                    raise RuntimeError("Takes too long for checkpoint")

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

            start = time.clock()
            next_state, reward, done, info = self.env.step(action)
            end = time.clock()
            if end - start > 0.1:
                print("Takes too long for step")
                raise RuntimeError("Takes too long for step")

            if self.enable_record and not info["unchanged"]:
                self.level_file_exporter.record_next(self.env.viewParser, info["action_for_viewer"])

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
        else:
            raise NotImplementedError()

    def get_action_type(self):
        return self.action_mode

    def check_point(self):
        if self.env_type == "HappyElimination":
            return self.env.check_point()
        else:
            raise NotImplementedError("Env does not support check_point")

    def load_check_point(self):
        if self.env_type == "HappyElimination":
            return self.env.load_check_point()
        else:
            raise NotImplementedError("Env does not support check_point")

    def enable_concentration_learning(self, progress, count, cool_down):
        if self.cool_down > 0:
            return False

        if self.env.need_concentration_learning():
            self.progress_condition = progress
            self.scheduled_count = count
            self.scheduled_cooldown = cool_down
