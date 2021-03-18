import numpy as np


class SumTree():
    def __init__(self, max_buffer_length):
        self.max_buffer_length = max_buffer_length

        self.node_buffer = np.zeros([max_buffer_length], dtype = np.int32)

        self.node_capacity = np.zeros([max_buffer_length * 2 - 1], dtype = np.float32)

        self.curr_buffer_length = 0

        self.write_pos = 0

        self.node_backhock = dict()

    def add_item(self, item, capacity):
        capacity = abs(capacity)

        self.node_buffer[self.write_pos] = item

        self.update_capacity(self.write_pos, capacity)

        self.write_pos = (self.write_pos + 1) % self.max_buffer_length

        if self.curr_buffer_length < self.max_buffer_length:
            self.curr_buffer_length += 1

    def update_node_capacity(self, batch, capacities):
        for idx, capacity in zip(batch, capacities):
            if idx in self.node_backhock:
                hocked_idx = self.node_backhock[idx]
                self.update_capacity(hocked_idx, capacity)

    def sample(self, batch_size):
        batch_capacity = np.random.random([batch_size]) * self.node_capacity[0]

        batch = []
        importance_sampling_factors = []

        for capacity in batch_capacity:
            idx, node_capacity = self.find_sample(capacity)
            batch.append(idx)
            importance_sampling_factor = self.node_capacity[0] / \
                self.curr_buffer_length / (node_capacity + 1e-4)
            importance_sampling_factor = min(importance_sampling_factor, 10.0)
            importance_sampling_factors.append(importance_sampling_factor)

        return batch, importance_sampling_factors

    def update_capacity(self, pos, capacity):
        pos = pos + self.max_buffer_length - 1

        delta = capacity - self.node_capacity[pos]

        while pos >= 0:
            self.node_capacity[pos] += delta

            pos = (pos - 1) // 2

    def find_sample(self, capacity, idx = 0):
        if idx >= self.max_buffer_length - 1:
            idx = idx - self.max_buffer_length + 1
            self.node_backhock[self.node_buffer[idx]] = idx
            return self.node_buffer[idx], self.node_capacity[idx]
        else:
            left = 2 * idx + 1
            right = left + 1

            if capacity < self.node_capacity[left]:
                return self.find_sample(capacity, idx = left)
            else:
                return self.find_sample(capacity - self.node_capacity[left], idx = right)
