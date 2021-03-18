import torch
import numpy as np
import warnings
import random
from copy import deepcopy

from mems.Memory import Memory


class HierarchyEliteStepMemory(Memory):
    def __init__(self, max_buffer_size, device = None):
        items_name = ["state", "action1", "action2", "reward1", "reward2",
                      "next_state", "done1", "done2", "score", "entrance"]
        max_sampling_count = 1000

        super(HierarchyEliteStepMemory, self).__init__(items_name,
                                              max_buffer_size = max_buffer_size,
                                              max_sampling_count = max_sampling_count,
                                              device = device)

        # Temporary items
        self.buffer_t = dict()
        self.buffer_t["state"] = None
        self.buffer_t["action1"] = None
        self.buffer_t["action2"] = None
        self.buffer_t["reward1"] = None
        self.buffer_t["reward2"] = None
        self.buffer_t["next_state"] = None
        self.buffer_t["done1"] = None
        self.buffer_t["done2"] = None
        self.buffer_t["score"] = None
        self.buffer_t["entrance"] = 0

        # Reference score
        self.ref_score = -10.0

        # Count threshold
        self.count_threshold = 100

    def append(self, state, action1, action2, reward1, reward2, done1, done2, score, clear_state = False):
        if self.buffer_t["state"] is not None:
            self.buffer_t["next_state"] = deepcopy(state)

            # Append to memory
            super(HierarchyEliteStepMemory, self).append(self.buffer_t)

        self.buffer_t["state"] = deepcopy(state)
        self.buffer_t["action1"] = deepcopy(action1)
        self.buffer_t["reward1"] = deepcopy(reward1)
        self.buffer_t["done1"] = deepcopy(done1)
        self.buffer_t["action2"] = deepcopy(action2)
        self.buffer_t["reward2"] = deepcopy(reward2)
        self.buffer_t["done2"] = deepcopy(done2)
        self.buffer_t["score"] = deepcopy(score)

        if clear_state:
            self.buffer_t["state"] = None

    def set_ref_score(self, ref_score):
        self.ref_score = ref_score

    def sample(self, batch_size, to_tensor = True):
        batch = super(HierarchyEliteStepMemory, self).sample(batch_size, to_tensor)

        if batch is None:
            return None, None, None, None, None, None, None, None
        else:
            return (batch["state"], batch["action1"], batch["action2"],
                    batch["reward1"], batch["reward2"],
                    batch["next_state"], batch["done1"], batch["done2"])

    def sampling_condition(self, idx):
        return self.buffers["score"][idx] > self.ref_score

    def super_sample(self, batch_size, to_tensor = True, return_idxs = False, sample_recent = None):
        batch_idxs = self.super_sample_random_idxs(batch_size, sample_recent = sample_recent)

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

    def super_sample_random_idxs(self, batch_size, sample_recent = None):
        if self.buffer_size == 0:
            return None
        elif batch_size > self.buffer_size:
            warnings.warn("Batch size is bigger than buffer size, be careful of over-sampling.")

            batch_idxs = np.random.random_integers(0, self.buffer_size - 1, size = batch_size)
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

            if self.buffers["entrance"][batch_idx] > self.count_threshold:
                self.buffers["entrance"][batch_idx] -= 1

                count = 0
                while (not self.sampling_condition(batch_idx)) and count < self.max_sampling_count:
                    batch_idx = np.random.randint(min_idx, self.buffer_size)
                    count += 1

                    if count >= self.max_sampling_count:
                        return None
                        # raise RuntimeError("No valid sample in replay memory, please check implementation.")

                batch_idxs[i] = batch_idx

            self.buffers["entrance"][batch_idx] += 1

        return batch_idxs


