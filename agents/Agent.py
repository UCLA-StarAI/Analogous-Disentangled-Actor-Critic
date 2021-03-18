import torch
import numpy as np


class Agent():
    def __init__(self, state_shape, action_type, action_params, args, device = None):
        self.state_shape = state_shape
        self.action_type = action_type
        self.action_params = action_params

        self.gamma = args.gamma
        self.device = device

        self.required_training_mode = "off_policy"

    def train_step(self):
        raise NotImplementedError()

    def action(self, state, mode = "train"):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def observe(self, state, action, reward, done):
        raise NotImplementedError()

    def to(self, device = None):
        raise NotImplementedError()

    def load(self, folder_path):
        raise NotImplementedError()

    def save(self, folder_path):
        raise NotImplementedError()

    def to_tensor(self, var, requires_grad = False):
        if self.device is not None:
            return torch.tensor(var, dtype = torch.float32,
                                requires_grad = requires_grad).to(self.device)
        else:
            return torch.tensor(var, dtype = torch.float32,
                                requires_grad = requires_grad)

    def to_numpy(self, var):
        return var.detach().cpu().numpy()

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def array_slice_2d(self, array, idxs1, idxs2):
        if isinstance(array, np.ndarray):
            assert len(np.shape(array)) == 2

            raise NotImplementedError()
        else:
            size = array.size()
            assert len(size) == 2

            array = array.view(-1)
            # print(idxs1.type(), idxs2.type(), idxs1.size(), idxs2.size(), size[1])
            idxs1 = idxs1 * size[1] + idxs2.view(-1)

            array = array[idxs1]

        return array

    def array_slice_3d(self, array, idxs1, idxs2):
        if isinstance(array, np.ndarray):
            assert len(np.shape(array)) == 3

            raise NotImplementedError()
        else:
            size = array.size()
            assert len(size) == 3

            array = array.view(-1, size[2])
            idxs1 = idxs1 * size[1] + idxs2.view(-1)

            array = array[idxs1, :]

        return array
