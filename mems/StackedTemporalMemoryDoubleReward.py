import numpy as np
import torch
from copy import deepcopy


class StackedTemporalMemoryDoubleReward():
    def __init__(self, env_num, mem_step_len, state_shape, action_type,
                 action_params, device = None):
        self.env_num = env_num
        self.mem_step_len = mem_step_len
        self.state_shape = state_shape
        self.action_type = action_type
        self.action_params = action_params
        self.device = device

        # Current idx
        self.step_idx = 0

        # Temporal buffers
        if isinstance(state_shape, dict):
            self.state_buffer = dict()
            self.state_buffer["Img"] = np.zeros([mem_step_len + 1, env_num, *(state_shape["Img"])], dtype = np.float32)
            self.state_buffer["Vec"] = np.zeros([mem_step_len + 1, env_num, *(state_shape["Vec"])], dtype=np.float32)
        else:
            self.state_buffer = np.zeros([mem_step_len + 1, env_num, *state_shape], dtype = np.float32)

        if action_type == "Discrete":
            self.action_buffer = np.zeros([mem_step_len + 1, env_num, 1], dtype = np.int64)
        elif action_type == "Continuous":
            self.action_buffer = np.zeros([mem_step_len + 1, env_num, action_params["dims"]], dtype = np.float32)
        else:
            raise NotImplementedError()

        self.reward_buffer1 = np.zeros([mem_step_len + 1, env_num, 1], dtype = np.float32)
        self.reward_buffer2 = np.zeros([mem_step_len + 1, env_num, 1], dtype=np.float32)
        self.done_buffer = np.zeros([mem_step_len + 1, env_num, 1], dtype = np.float32)

    def append(self, state, action, reward1, reward2, done):
        if self.step_idx > self.mem_step_len:
            raise RuntimeError()
        if isinstance(self.state_shape, dict):
            img_state = np.array([item[0] for item in state])
            vec_state = np.array([item[1] for item in state])
            self.state_buffer["Img"][self.step_idx, :, :, :, :] = deepcopy(img_state)
            self.state_buffer["Vec"][self.step_idx, :, :] = deepcopy(vec_state)
        elif len(self.state_shape) == 1:
            self.state_buffer[self.step_idx, :, :] = deepcopy(state)
        elif len(self.state_shape) == 3:
            self.state_buffer[self.step_idx, :, :, :, :] = deepcopy(state)
        else:
            raise NotImplementedError()

        if self.action_type == "Discrete":
            self.action_buffer[self.step_idx, :, :] = np.expand_dims(deepcopy(action), axis = -1)
        elif self.action_type == "Continuous":
            self.action_buffer[self.step_idx, :, :] = deepcopy(action)
        else:
            raise NotImplementedError()

        self.reward_buffer1[self.step_idx, :, :] = np.expand_dims(deepcopy(reward1), axis = -1)
        self.reward_buffer2[self.step_idx, :, :] = np.expand_dims(deepcopy(reward2), axis = -1)
        self.done_buffer[self.step_idx, :, :] = np.expand_dims(deepcopy(done), axis = -1)

        self.step_idx += 1

    def sample(self, to_tensor = True):
        if self.step_idx < self.mem_step_len:
            raise RuntimeError()

        if isinstance(self.state_buffer, dict):
            img_state_batch = self.state_buffer["Img"].copy()
            vec_state_batch = self.state_buffer["Vec"].copy()
        else:
            state_batch = self.state_buffer.copy()
        action_batch = self.action_buffer[:-1, :, :].copy()
        reward_batch1 = self.reward_buffer1[:-1, :, :].copy()
        reward_batch2 = self.reward_buffer2[:-1, :, :].copy()
        done_batch = self.done_buffer[:-1, :, :].copy()

        if to_tensor:
            if self.device is not None:
                if isinstance(self.state_buffer, dict):
                    img_state_batch = torch.tensor(img_state_batch, dtype = torch.float32).to(self.device)
                    vec_state_batch = torch.tensor(vec_state_batch, dtype = torch.float32).to(self.device)
                else:
                    state_batch = torch.tensor(state_batch, dtype = torch.float32).to(self.device)
                if self.action_type == "Discrete":
                    action_batch = torch.tensor(action_batch, dtype = torch.int64).to(self.device)
                else:
                    action_batch = torch.tensor(action_batch, dtype = torch.float32).to(self.device)
                reward_batch1 = torch.tensor(reward_batch1, dtype = torch.float32).to(self.device)
                reward_batch2 = torch.tensor(reward_batch2, dtype=torch.float32).to(self.device)
                done_batch = torch.tensor(done_batch, dtype = torch.float32).to(self.device)
            else:
                if isinstance(self.state_buffer, dict):
                    img_state_batch = torch.tensor(img_state_batch, dtype = torch.float32)
                    vec_state_batch = torch.tensor(vec_state_batch, dtype = torch.float32)
                else:
                    state_batch = torch.tensor(state_batch, dtype = torch.float32)
                if self.action_type == "Discrete":
                    action_batch = torch.tensor(action_batch, dtype = torch.int64)
                else:
                    action_batch = torch.tensor(action_batch, dtype = torch.float32)
                reward_batch1 = torch.tensor(reward_batch1, dtype = torch.float32)
                reward_batch2 = torch.tensor(reward_batch2, dtype=torch.float32)
                done_batch = torch.tensor(done_batch, dtype = torch.float32)

        # Reset buffers
        self.step_idx = 1

        if isinstance(self.state_shape, dict):
            self.state_buffer["Img"][0, :, :, :, :] = self.state_buffer["Img"][-1, :, :, :, :]
            self.state_buffer["Vec"][0, :, :] = self.state_buffer["Vec"][-1, :, :]
        elif len(self.state_shape) == 1:
            self.state_buffer[0, :, :] = self.state_buffer[-1, :, :]
        elif len(self.state_shape) == 3:
            self.state_buffer[0, :, :, :, :] = self.state_buffer[-1, :, :, :, :]
        else:
            raise NotImplementedError()

        self.action_buffer[0, :, :] = self.action_buffer[-1, :, :]
        self.reward_buffer1[0, :, :] = self.reward_buffer1[-1, :, :]
        self.reward_buffer2[0, :, :] = self.reward_buffer2[-1, :, :]
        self.done_buffer[0, :, :] = self.done_buffer[-1, :, :]

        if isinstance(self.state_buffer, dict):
            return (img_state_batch, vec_state_batch), action_batch, reward_batch1, reward_batch2, done_batch
        else:
            return state_batch, action_batch, reward_batch1, reward_batch2, done_batch

    def ready_for_training(self):
        if self.step_idx > self.mem_step_len:
            return True
        else:
            return False
