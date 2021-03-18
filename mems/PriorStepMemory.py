import numpy as np
import torch
from copy import deepcopy

from utils.SumTree import SumTree


class PriorStepMemory():
    def __init__(self, max_buffer_size, device = None, td_error = None):
        self.max_buffer_size = max_buffer_size

        # Buffers
        self.state_buffer = np.zeros(max_buffer_size, dtype = np.object)
        self.action_buffer = np.zeros(max_buffer_size, dtype = np.object)
        self.reward_buffer = np.zeros(max_buffer_size, dtype = np.object)
        self.next_state_buffer = np.zeros(max_buffer_size, dtype = np.object)
        self.done_buffer = np.zeros(max_buffer_size, dtype = np.object)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action"] = None
        self.buffer_t["reward"] = None
        self.buffer_t["next_state"] = None
        self.buffer_t["done"] = None

        # Sum Tree
        self.sumTree = SumTree(max_buffer_size)

        # Device
        self.device = device

        # Write pos
        self.write_pos = 0

        # TD-error
        self.td_error = td_error

        # Last batch idxs
        self.last_batch_idxs = []

    def append(self, state, action, reward, done):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            capacity = self.td_error(
                self.buffer_t["state"],
                self.buffer_t["action"],
                self.buffer_t["reward"],
                self.buffer_t["next_state"],
                self.buffer_t["done"]
            )

            self.state_buffer[self.write_pos] = self.buffer_t["state"]
            self.action_buffer[self.write_pos] = self.buffer_t["action"]
            self.reward_buffer[self.write_pos] = self.buffer_t["reward"]
            self.next_state_buffer[self.write_pos] = self.buffer_t["next_state"]
            self.done_buffer[self.write_pos] = self.buffer_t["done"]

            self.sumTree.add_item(self.write_pos, capacity)

            self.write_pos = (self.write_pos + 1) % self.max_buffer_size

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action"] = deepcopy(action)
        self.buffer_t["reward"] = deepcopy(reward)
        self.buffer_t["done"] = deepcopy(done)

    def sample(self, batch_size, to_tensor = True):
        batch_idxs, importance_sampling_factors = self.sumTree.sample(batch_size)
        self.last_batch_idxs = batch_idxs

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        importance_sampling_factor_batch = []

        for batch_idx, importance_sampling_factor in zip(batch_idxs, importance_sampling_factors):
            state_batch.append(self.state_buffer[batch_idx])
            action_batch.append(self.action_buffer[batch_idx])
            reward_batch.append(self.reward_buffer[batch_idx])
            next_state_batch.append(self.next_state_buffer[batch_idx])
            done_batch.append(1.0 if self.done_buffer[batch_idx] else 0.0)
            importance_sampling_factor_batch.append(importance_sampling_factor)

        state_batch = np.stack(state_batch, axis = 0)
        action_batch = np.stack(action_batch, axis = 0)
        reward_batch = np.stack(reward_batch, axis = 0)
        next_state_batch = np.stack(next_state_batch, axis = 0)
        done_batch = np.stack(done_batch, axis = 0)
        importance_sampling_factor_batch = np.stack(importance_sampling_factor_batch, axis = 0)

        if to_tensor:
            if self.device is not None:
                state_batch = torch.tensor(state_batch, dtype = torch.float32).to(self.device)
                action_batch = torch.tensor(action_batch, dtype = torch.float32).to(self.device)
                reward_batch = torch.tensor(reward_batch, dtype = torch.float32).to(self.device)
                next_state_batch = torch.tensor(next_state_batch, dtype = torch.float32).to(self.device)
                done_batch = torch.tensor(done_batch, dtype = torch.float32).to(self.device)
                importance_sampling_factor_batch = torch.tensor(importance_sampling_factor_batch,
                                                                dtype = torch.float32).to(self.device)
            else:
                state_batch = torch.tensor(state_batch, dtype = torch.float32)
                action_batch = torch.tensor(action_batch, dtype = torch.float32)
                reward_batch = torch.tensor(reward_batch, dtype = torch.float32)
                next_state_batch = torch.tensor(next_state_batch, dtype = torch.float32)
                done_batch = torch.tensor(done_batch, dtype = torch.float32)
                importance_sampling_factor_batch = torch.tensor(importance_sampling_factor_batch,
                                                                dtype = torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, importance_sampling_factor_batch

    def update_node_capacity(self, td_error):
        self.sumTree.update_node_capacity(self.last_batch_idxs, td_error)
