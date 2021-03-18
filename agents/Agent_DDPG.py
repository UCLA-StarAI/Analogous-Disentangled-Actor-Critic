import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from agents.Agent import Agent
from networks.FCNet import FCNet

from mems.StepMemory import StepMemory

from utils.OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess


class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, action_range):
        super(Actor, self).__init__()

        self.input_len = len(state_shape)
        if self.input_len == 1:
            self.network = FCNet(state_shape[0], [200, 32, action_dim], activation_func = "Sigmoid")
        else:
            raise NotImplementedError()

        self.action_range = action_range

    def forward(self, state):
        if self.input_len == 1:
            if len(state.size()) == 1:
                state = state.unsqueeze(0)
        else:
            raise NotImplementedError()

        action = self.network(state)
        action = action * (self.action_range[1] - self.action_range[0]) + self.action_range[0]

        return action


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Critic, self).__init__()

        self.input_len = len(state_shape)
        if self.input_len == 1:
            self.state_encoder = FCNet(state_shape[0], [64], activation_func = "ReLU")
        else:
            raise NotImplementedError()

        self.action_encoder = FCNet(action_dim, [32], activation_func = "ReLU")

        self.feature_encoder = FCNet(64 + 32, [64, 32, 1], activation_func = "None")

    def forward(self, state, action):
        if self.input_len == 1:
            if len(state.size()) == 1:
                state = state.unsqueeze(0)
                action = action.unsqueeze(0)
        else:
            raise NotImplementedError()

        state_feature = self.state_encoder(state)
        action_feature = self.action_encoder(action)

        Q_value = self.feature_encoder(torch.cat((state_feature, action_feature), dim = 1))

        return Q_value


class Agent_DDPG(Agent):
    def __init__(self, state_shape, action_type, action_params, args, device = None, tau = 1e-3):
        assert action_type == "Continuous", "DDPG can only handle environment with " + \
                                            "continuous action space."

        super(Agent_DDPG, self).__init__(state_shape, action_type,
                                         action_params, args, device)

        # Networks
        self.actor = Actor(state_shape, action_params["dims"], action_params["range"])
        self.actor_target = Actor(state_shape, action_params["dims"], action_params["range"])
        self.critic = Critic(state_shape, action_params["dims"])
        self.critic_target = Critic(state_shape, action_params["dims"])

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = 1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = 1e-3)

        # Random process
        self.random_process = OrnsteinUhlenbeckProcess(size = action_params["dims"],
                                                       theta = 0.15,
                                                       mu = 0.0,
                                                       sigma = 0.2)

        # Memory
        self.memory = StepMemory(args.max_buffer_size, device = device)

        # Parameters
        self.training_batch_size = args.training_batch_size
        self.action_range = action_params["range"]
        self.tau = tau
        self.epsilon_start = 1.0
        self.epsilon_end = 0.0
        self.epsilon_n = 1000000
        self.epsilon = 1.0

        # Loss function
        self.mseLoss = nn.MSELoss()

    def train_step(self):
        state_batch, action_batch, reward_batch, \
            next_state_batch, done_batch = self.memory.sample(self.training_batch_size)

        with torch.no_grad():
            # Q(s',pi(s'))
            next_Q_values = self.critic_target(
                next_state_batch,
                self.actor_target(next_state_batch)
            )

            # target of Q(s,a)
            target_Q_batch = reward_batch.unsqueeze(-1) + self.gamma * (1.0 - done_batch.unsqueeze(-1)) * next_Q_values

        # Critic update
        self.critic.zero_grad()

        Q_batch = self.critic(state_batch, action_batch)
        value_loss = self.mseLoss(Q_batch, target_Q_batch)
        value_loss.backward()

        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic(
            state_batch,
            self.actor(state_batch)
        ).mean()
        policy_loss.backward()

        self.actor_optim.step()

        # Target network update
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

    def action(self, state, mode = "train"):
        if mode == "train":
            action = self.to_numpy(
                self.actor_target(self.to_tensor(state).unsqueeze(0)).squeeze(0)
            )
            action += max(self.epsilon, self.epsilon_end) * self.random_process.sample()
            action = np.clip(action, self.action_range[0], self.action_range[1])
        elif mode == "test":
            action = self.to_numpy(
                self.actor_target(self.to_tensor(state).unsqueeze(0)).squeeze(0)
            )
        else:
            raise NotImplementedError()

        return action

    def train(self):
        self.actor_target.train()

    def eval(self):
        self.actor_target.eval()

    def reset(self):
        self.random_process.reset_states()

    def observe(self, state, action, reward, done):
        self.memory.append(state, action, reward, done)

    def to(self, device = None):
        if device is None:
            self.actor.to(self.device)
            self.actor_target.to(self.device)
            self.critic.to(self.device)
            self.critic_target.to(self.device)
        else:
            self.actor.to(device)
            self.actor_target.to(device)
            self.critic.to(device)
            self.critic_target.to(device)

    def load(self, folder_path):
        save_file_path = os.path.join(folder_path, "models.pt")

        if not os.path.exists(save_file_path):
            return False

        checkpoint = torch.load(save_file_path, map_location = 'cpu')
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim"])

        return True

    def save(self, folder_path):
        save_file_path = os.path.join(folder_path, "models.pt")

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        torch.save(
            {'actor': self.actor.state_dict(),
             'actor_target': self.actor_target.state_dict(),
             'actor_optim': self.actor_optim.state_dict(),
             'critic': self.critic.state_dict(),
             'critic_target': self.critic_target.state_dict(),
             'critic_optim': self.critic_optim.state_dict()},
            save_file_path
        )
