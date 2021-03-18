import numpy as np
import torch
from copy import deepcopy


class HierarchicalStackedTemporalMemory_v1():
    def __init__(self, env_num, mem_step_len, state_shape, action_type,
                 action_params, target_dim, target_embedding_model, device = None):
        self.env_num = env_num
        self.mem_step_len = mem_step_len
        self.state_shape = state_shape
        self.target_dim = target_dim
        self.action_type = action_type
        self.action_params = action_params
        self.target_embedding_model = target_embedding_model
        self.device = device

        # Current idx
        self.step_idx = 0

        # Temporal buffers
        self.state_buffer = np.zeros([mem_step_len + 1, env_num, *state_shape], dtype = np.float32)
        self.target_buffer = np.zeros([mem_step_len + 1, env_num, target_dim], dtype = np.float32)
        if action_type == "Discrete":
            self.action_buffer = np.zeros([mem_step_len + 1, env_num, 1], dtype = np.int64)
        elif action_type == "Continuous":
            self.action_buffer = np.zeros([mem_step_len + 1, env_num, action_params["dim"]], dtype = np.float32)
        else:
            raise NotImplementedError()
        self.actor_reward_buffer = np.zeros([mem_step_len, env_num, 1], dtype = np.float32)
        self.reward_buffer = np.zeros([mem_step_len + 1, env_num, 1], dtype = np.float32)
        self.done_buffer = np.zeros([mem_step_len + 1, env_num, 1], dtype = np.float32)

    def append(self, state, target, action, reward, done):
        if self.step_idx > self.mem_step_len:
            raise RuntimeError()

        if len(self.state_shape) == 1:
            self.state_buffer[self.step_idx, :, :] = deepcopy(state)
        elif len(self.state_shape) == 3:
            self.state_buffer[self.step_idx, :, :, :, :] = deepcopy(state)
        else:
            raise NotImplementedError()
        target = target.detach().cpu().numpy()
        self.target_buffer[self.step_idx, :, :] = deepcopy(target)
        if self.action_type == "Discrete":
            self.action_buffer[self.step_idx, :, :] = np.expand_dims(deepcopy(action), axis = -1)
        elif self.action_type == "Continuous":
            self.action_buffer[self.step_idx, :, :] = deepcopy(action)
        else:
            raise NotImplementedError()
        self.reward_buffer[self.step_idx, :, :] = np.expand_dims(deepcopy(reward), axis = -1)
        self.done_buffer[self.step_idx, :, :] = np.expand_dims(deepcopy(done), axis = -1)

        # actor_reward_buffer
        if self.step_idx >= 1:
            with torch.no_grad():
                if len(self.state_shape) == 1:
                    last_target_embedding = self.target_embedding_model(
                        torch.tensor(self.state_buffer[self.step_idx - 1, :, :])
                    )
                    curr_target_embedding = self.target_embedding_model(
                        torch.tensor(self.state_buffer[self.step_idx, :, :])
                    )
                    target_embedding = torch.tensor(self.target_buffer[self.step_idx - 1, :, :], dtype = torch.float32)
                    last_dist = torch.pow(last_target_embedding -
                                          target_embedding, 2).sum(dim = 1).sqrt()
                    curr_dist = torch.pow(curr_target_embedding -
                                          target_embedding, 2).sum(dim = 1).sqrt()
                    self.actor_reward_buffer[self.step_idx - 1, :, 0] = last_dist - curr_dist
                else:
                    raise NotImplementedError()

        self.step_idx += 1

    def sample(self, to_tensor = True):
        if self.step_idx < self.mem_step_len:
            raise RuntimeError()

        state_batch = self.state_buffer.copy()
        target_batch = self.target_buffer[:-1, :, :].copy()
        actor_reward_batch = self.actor_reward_buffer.copy()
        action_batch = self.action_buffer[:-1, :, :].copy()
        reward_batch = self.reward_buffer[:-1, :, :].copy()
        done_batch = self.done_buffer[:-1, :, :].copy()

        if to_tensor:
            if self.device is not None:
                state_batch = torch.tensor(state_batch, dtype = torch.float32).to(self.device)
                target_batch = torch.tensor(target_batch, dtype = torch.float32).to(self.device)
                if self.action_type == "Discrete":
                    action_batch = torch.tensor(action_batch, dtype = torch.int64).to(self.device)
                else:
                    action_batch = torch.tensor(action_batch, dtype = torch.float32).to(self.device)
                    actor_reward_batch = torch.tensor(actor_reward_batch, dtype = torch.float32).to(self.device)
                actor_reward_batch = torch.tensor(actor_reward_batch, dtype = torch.float32).to(self.device)
                reward_batch = torch.tensor(reward_batch, dtype = torch.float32).to(self.device)
                done_batch = torch.tensor(done_batch, dtype = torch.float32).to(self.device)
            else:
                state_batch = torch.tensor(state_batch, dtype = torch.float32)
                target_batch = torch.tensor(target_batch, dtype = torch.float32)
                if self.action_type == "Discrete":
                    action_batch = torch.tensor(action_batch, dtype = torch.int64)
                else:
                    action_batch = torch.tensor(action_batch, dtype = torch.float32)
                actor_reward_batch = torch.tensor(actor_reward_batch, dtype = torch.float32)
                reward_batch = torch.tensor(reward_batch, dtype = torch.float32)
                done_batch = torch.tensor(done_batch, dtype = torch.float32)

        # Reset buffers
        self.step_idx = 1
        if len(self.state_shape) == 1:
            self.state_buffer[0, :, :] = self.state_buffer[-1, :, :]
        elif len(self.state_shape) == 3:
            self.state_buffer[0, :, :, :, :] = self.state_buffer[-1, :, :, :, :]
        else:
            raise NotImplementedError()
        self.action_buffer[0, :, :] = self.action_buffer[-1, :, :]
        self.target_buffer[0, :, :] = self.target_buffer[-1, :, :]
        self.reward_buffer[0, :, :] = self.reward_buffer[-1, :, :]
        self.done_buffer[0, :, :] = self.reward_buffer[-1, :, :]

        return state_batch, target_batch, action_batch, actor_reward_batch, reward_batch, done_batch

    def ready_for_training(self):
        if self.step_idx > self.mem_step_len:
            return True
        else:
            return False
