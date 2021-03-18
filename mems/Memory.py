import torch
import numpy as np
import random
import warnings
from heapq import heappush, heappop, nsmallest
from copy import deepcopy

from RingBuffer import RingBuffer


# Overrideable function: sampling_condition()
# Function need to call super: sample(), append()
class Memory():
    def __init__(self, items_name, max_buffer_size, max_sampling_count = 10000,
                 device = None, use_priorized_heap = False):
        self.items_name = items_name
        self.max_buffer_size = max_buffer_size
        self.buffers = dict()
        self.items_shape = dict()
        self.batch_template = dict()

        # Register buffers
        for item_name in items_name:
            self.buffers[item_name] = RingBuffer(max_buffer_size)

        # Empty item dictionary & initialize items_shape
        for item_name in items_name:
            self.batch_template[item_name] = []
            self.items_shape[item_name] = None

        # Replay buffer size
        self.buffer_size = 0

        # Max sampling count
        self.max_sampling_count = max_sampling_count

        # Device
        self.device = device

        # Prorized heap
        self.use_priorized_heap = use_priorized_heap
        if use_priorized_heap:
            self.heap_start_index = 0
            self.curr_heap_idx = 0
            self.curr_heap_size = 0
            self.prior_heap = []
            self.extracted_heap_items = []

    def append(self, items_dict, prior_heap_key = 0.0):
        for item_name in self.items_name:
            if item_name not in items_dict:
                raise KeyError("The item_dict appended to replay buffer is missing key %s." % item_name)
            if self.items_shape[item_name] is None:
                self.items_shape[item_name] = np.shape(items_dict[item_name])
            elif self.items_shape[item_name] != np.shape(items_dict[item_name]):
                raise ValueError("Buffer %s with shape %s received an item with shape %s." %
                                 (item_name, str(self.items_shape[item_name]), str(np.shape(items_dict[item_name]))))

            self.buffers[item_name].append(deepcopy(items_dict[item_name]))

        # Maintain heap, if needed
        if self.use_priorized_heap:
            heappush(self.prior_heap, (prior_heap_key, self.curr_heap_idx))
            self.curr_heap_idx = (self.curr_heap_idx + 1) % self.max_buffer_size
            self.curr_heap_size += 1
            if self.buffer_size > self.max_buffer_size:
                self.heap_start_index = (self.heap_start_index + 1) % self.max_buffer_size
            if self.curr_heap_size > 5 * self.max_buffer_size:
                self.heap_reduce()

        # Maintain buffer size
        self.buffer_size += 1
        if self.buffer_size > self.max_buffer_size:
            self.buffer_size = self.max_buffer_size

    def full(self):
        return self.buffer_size == self.max_buffer_size

    def sample(self, batch_size, to_tensor = True, return_idxs = False, sample_recent = None):
        batch_idxs = self.sample_random_idxs(batch_size, sample_recent = sample_recent)

        if batch_idxs is None:
            return None

        batch = dict()
        for item_name in self.items_name:
            batch[item_name] = []
            for batch_idx in batch_idxs:
                batch[item_name].append(deepcopy(self.buffers[item_name][batch_idx]))

        for item_name in self.items_name:
            if isinstance(batch[item_name][0], list):
                curr_batch = []
                for i in range(len(batch[item_name][0])):
                    curr_batch.append(np.array([item[i] for item in batch[item_name]], dtype = np.float32))
                batch[item_name] = curr_batch
            else:
                batch[item_name] = np.array(batch[item_name], dtype = np.float32)

        if to_tensor:
            if self.device is not None:
                for item_name in self.items_name:
                    batch[item_name] = torch.tensor(batch[item_name], dtype = torch.float32).to(self.device)
            else:
                for item_name in self.items_name:
                    batch[item_name] = torch.tensor(batch[item_name], dtype = torch.float32)

        if return_idxs:
            return batch, self.buffer_size - batch_idxs
        else:
            return batch

    def sample_random_idxs(self, batch_size, sample_recent = None):
        if self.buffer_size == 0:
            return None
        elif batch_size > self.buffer_size:
            warnings.warn("Batch size is bigger than buffer size, be careful of over-sampling.")

            batch_idxs = np.random.random_integers(0, self.buffer_size - 1, size = batch_size)
        elif self.use_priorized_heap:
            self.extracted_heap_items.clear()
            batch_idxs = []
            for _ in range(batch_size):
                item = heappop(self.prior_heap)
                batch_idxs.append(item[1] + self.heap_start_index)
                self.extracted_heap_items.append(item)
        else:
            if sample_recent is None:
                r = range(0, self.buffer_size)
            else:
                r = range(max(0, self.buffer_size - sample_recent), self.buffer_size)

            batch_idxs = random.sample(r, batch_size)

        batch_idxs = np.array(batch_idxs, dtype = np.int64)

        min_idx = 0 if sample_recent is None else max(0, self.buffer_size - sample_recent)

        # Remove and correct samples that can not be sampled.
        for i, batch_idx in enumerate(batch_idxs):
            if not self.sampling_condition(batch_idx):
                count = 0
                while (not self.sampling_condition(batch_idx)) and count < self.max_sampling_count:
                    batch_idx = np.random.randint(min_idx, self.buffer_size)
                    count += 1

                if count >= self.max_sampling_count:
                    return None
                    # raise RuntimeError("No valid sample in replay memory, please check implementation.")

                batch_idxs[i] = batch_idx

        return batch_idxs

    def sampling_condition(self, idx):
        raise NotImplementedError("Must override member function sampling_condition.")

    def heap_reduce(self):
        n_smallest_items = nsmallest(self.max_buffer_size, self.prior_heap)
        self.prior_heap = []
        self.curr_heap_size = self.max_buffer_size
        for item in n_smallest_items:
            heappush(self.prior_heap, item)

    def heap_refill(self, prior_heap_keys):
        for heap_item, key in zip(self.extracted_heap_items, prior_heap_keys):
            heappush(self.prior_heap, (key, heap_item[1]))
