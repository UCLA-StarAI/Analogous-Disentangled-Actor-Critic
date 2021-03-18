import numpy as np
import torch
import torch.nn as nn


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class NegSig(nn.Module):
    def __init__(self):
        super(NegSig, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x) * -10


def fanin_init(size, fanin = None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class FCNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_func = "None",
                 internal_activation_func = "ReLU"):
        super(FCNet, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        self.linear_modules = []

        last_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            unit = nn.Linear(last_size, hidden_size)
            if i == len(hidden_sizes) - 1:
                unit.weight.data.uniform_(-1e-3, 1e-3)
            else:
                unit.weight.data = fanin_init(unit.weight.data.size())

            self.add_module("unit" + str(i), unit)
            self.linear_modules.append(unit)

            last_size = hidden_size

        if internal_activation_func == "ReLU":
            self.ReLU = nn.ReLU(inplace = True)
        elif internal_activation_func == "ELU":
            self.ReLU = nn.ELU(alpha = 1.0)
        else:
            raise NotImplementedError()

        if activation_func == "ReLU":
            self.finishing_activ = nn.ReLU(inplace = True)
        elif activation_func == "ELU":
            self.finishing_activ = nn.ELU(alpha = 1.0)
        elif activation_func == "Sigmoid":
            self.finishing_activ = nn.Sigmoid()
        elif activation_func == "Tanh":
            self.finishing_activ = nn.Tanh()
        elif activation_func == "Softmax":
            self.finishing_activ = nn.Softmax(dim = 1)
        elif activation_func == "Softplus":
            self.finishing_activ = nn.Softplus()
        elif activation_func == "NegSig":
            self.finishing_activ = NegSig()
        elif activation_func == "None":
            self.finishing_activ = EmptyModule()
        else:
            raise NotImplementedError()

        self.activation_func = activation_func

    def forward(self, x):
        for i, unit in enumerate(self.linear_modules):
            x = unit(x)
            if i != len(self.linear_modules) - 1:
                x = self.ReLU(x)

        if self.activation_func == "Softmax":
            x = self.finishing_activ(x + 1e-6)
        else:
            x = self.finishing_activ(x)

        return x
