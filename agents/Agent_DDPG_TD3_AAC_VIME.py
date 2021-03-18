import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import math

from agents.Agent import Agent
from networks.FCNet import FCNet

from mems.StepMemoryDoubleReward import StepMemoryDoubleReward

from utils.MovingAvegCalculator import MovingAvegCalculator


class WorldModel(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(WorldModel, self).__init__()

        assert len(state_shape) == 1
        state_dim = state_shape[0]

        if state_dim > 32:
            factor = 2.0
        elif state_dim > 16:
            factor = 1.0
        elif state_dim > 8:
            factor = 0.6
        else:
            factor = 0.4

        self.state_feature_network = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, int(200 * factor)),
            nn.ReLU()
        )

        self.action_feature_network = nn.Sequential(
            nn.Linear(action_dim, 100),
            nn.ReLU()
        )

        self.env_feature_network_mean = nn.Sequential(
            nn.Linear(int(200 * factor) + 100, int(200 * factor)),
            nn.Tanh()
        )

        self.env_feature_network_std = nn.Sequential(
            nn.Linear(int(200 * factor) + 100, int(200 * factor))
        )

        self.softplus = nn.Softplus()

        self.state_generation_network = nn.Sequential(
            nn.Linear(int(200 * factor), state_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)

        self.mseLoss = nn.MSELoss()

        self.moving_aveg_calculator = MovingAvegCalculator(window_length = 5000)

    def forward(self, state, action):
        state_feature = self.state_feature_network(state)
        action_feature = self.action_feature_network(action)

        concat_feature = torch.cat((state_feature, action_feature), dim = 1)

        feature_mean = self.env_feature_network_mean(concat_feature)
        feature_std_raw = self.env_feature_network_mean(concat_feature)
        feature_std = self.softplus(feature_std_raw)

        feature = feature_mean + torch.normal(
            mean = torch.zeros_like(feature_mean), std = 1).to(state.device) * feature_std

        next_state = self.state_generation_network(feature)

        return next_state, feature_mean, feature_std, feature_std_raw

    def train_step(self, state, action, next_state):
        next_state_batch, feature_mean, feature_std, _ = self.forward(state, action)

        feature_mean_2 = feature_mean.pow(2)
        feature_std_2 = feature_std.pow(2)

        div_loss = -(1 + 2 * torch.log(feature_std) - feature_mean_2 - feature_std_2).mean()

        mse_loss = self.mseLoss(next_state_batch, next_state)

        loss = mse_loss + div_loss

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def get_extra_reward(self, state, action, next_state):
        with torch.no_grad():
            state_feature = self.state_feature_network(state)
            action_feature = self.action_feature_network(action)

            concat_feature = torch.cat((state_feature, action_feature), dim=1)

            feature_mean = self.env_feature_network_mean(concat_feature)
            feature_std_raw = self.env_feature_network_mean(concat_feature)
            feature_std = self.softplus(feature_std_raw)

        feature_mean.requires_grad = True
        feature_std_raw.requires_grad = True

        feature = feature_mean + torch.normal(
            mean = torch.zeros_like(feature_mean), std = 1).to(state.device) * self.softplus(feature_std_raw)

        next_state_pred = self.state_generation_network(feature)

        loss = self.mseLoss(next_state_pred, next_state)

        loss.backward()

        with torch.no_grad():
            feature_mean_grad = feature_mean.grad.data
            feature_std_grad = feature_std_raw.grad.data

            hessian_for_mean = feature_std.pow(2)
            hessian_for_std = torch.exp(2 * feature_std) * feature_std.pow(2) / \
                              (2 * torch.exp(feature_std_raw * 2))

            extra_reward = ((feature_mean_grad.pow(2) * hessian_for_mean).mean() +
                            (feature_std_grad.pow(2) * hessian_for_std).mean()) / 2

        extra_reward = extra_reward.detach().cpu().numpy()

        aveg_extra_reward = self.moving_aveg_calculator.add_number(extra_reward)[0]

        return min(max(extra_reward / aveg_extra_reward, 0.0), 3.0)


class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, action_range, rand_var_dim = 16):
        super(Actor, self).__init__()

        self.input_len = len(state_shape)
        if self.input_len == 1:
            self.state_encoder = FCNet(state_shape[0], [400, 300], activation_func = "ReLU")
            self.randomness_encoder = FCNet(300 + rand_var_dim, [100, action_dim], activation_func = "Sigmoid")
        else:
            raise NotImplementedError()

        self.action_range = action_range

        self.rand_var_dim = rand_var_dim

    def forward(self, state, mode = "Std"):
        if self.input_len == 1:
            if len(state.size()) == 1:
                state = state.unsqueeze(0)
        else:
            raise NotImplementedError()

        if mode == "Std":
            rand_var = torch.zeros([state.size(0), self.rand_var_dim], dtype = torch.float32).to(state.device)
        elif mode == "Ent":
            rand_var = torch.normal(
                mean = torch.zeros([state.size(0), self.rand_var_dim], dtype = torch.float32),
                std = 1
            ).to(state.device)
        else:
            raise NotImplementedError()

        state_feature = self.state_encoder(state)
        action = self.randomness_encoder(torch.cat((state_feature, rand_var), dim = 1))

        action = action * (self.action_range[1] - self.action_range[0]) + self.action_range[0]

        return action


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Critic, self).__init__()

        self.input_len = len(state_shape)
        if self.input_len == 1:
            self.state_encoder = FCNet(state_shape[0] + action_dim, [400], activation_func = "ReLU")
        else:
            raise NotImplementedError()

        self.action_encoder = FCNet(400 + action_dim, [300], activation_func = "ReLU")

        self.feature_encoder = FCNet(300, [1], activation_func = "None")

    def forward(self, state, action):
        if self.input_len == 1:
            if len(state.size()) == 1:
                state = state.unsqueeze(0)
                action = action.unsqueeze(0)
        else:
            raise NotImplementedError()

        state_feature = self.state_encoder(torch.cat((state, action), dim = 1))
        action_feature = self.action_encoder(torch.cat((state_feature, action), dim = 1))

        Q_value = self.feature_encoder(action_feature)

        return Q_value


class Agent_DDPG_TD3_AAC_VIME(Agent):
    def __init__(self, state_shape, action_type, action_params,
                 args, device = None, tau = 5e-3, store_replay_buffer = True):
        assert action_type == "Continuous", "DDPG_TD3 can only handle environment with " + \
                                            "continuous action space."

        super(Agent_DDPG_TD3_AAC_VIME, self).__init__(state_shape, action_type,
                                                      action_params, args, device)

        # Networks
        self.actor = Actor(state_shape, action_params["dims"], action_params["range"])
        self.actor_target = Actor(state_shape, action_params["dims"], action_params["range"])
        self.critic1 = Critic(state_shape, action_params["dims"])
        self.critic1_target = Critic(state_shape, action_params["dims"])
        self.critic2 = Critic(state_shape, action_params["dims"])
        self.critic2_target = Critic(state_shape, action_params["dims"])
        self.critic3 = Critic(state_shape, action_params["dims"])
        self.critic3_target = Critic(state_shape, action_params["dims"])
        self.critic4 = Critic(state_shape, action_params["dims"])
        self.critic4_target = Critic(state_shape, action_params["dims"])

        self.world_model = WorldModel(state_shape, action_params["dims"])

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic1_target, self.critic1)
        self.hard_update(self.critic2_target, self.critic2)
        self.hard_update(self.critic3_target, self.critic3)
        self.hard_update(self.critic4_target, self.critic4)

        # Optimizers
        self.actor_optim_std = optim.Adam(self.actor.parameters(), lr = 1e-3)
        self.actor_optim_ent = optim.Adam(self.actor.parameters(), lr = 1e-3)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr = 1e-3)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr = 1e-3)
        self.critic3_optim = optim.Adam(self.critic3.parameters(), lr = 1e-3)
        self.critic4_optim = optim.Adam(self.critic4.parameters(), lr = 1e-3)

        # Memory
        self.memory = StepMemoryDoubleReward(args.max_buffer_size, device = device)

        self.recent_memory = StepMemoryDoubleReward(1000, device = device)

        # Parameters
        self.training_batch_size = 100
        self.action_range = action_params["range"]
        self.tau = tau
        self.epsilon = 0.1

        # Balance_temperature
        self.start_balance_temperature = 2.0
        self.end_balance_temperature = 1.0
        self.balance_temperature = self.start_balance_temperature
        self.balance_temperature_n = 15000000

        # Other parameters
        self.K = 32
        self.sigma = min(math.sqrt(self.action_params["dims"]) / self.K, 0.1)

        # Loss function
        self.mseLoss = nn.MSELoss()

        # Train step count
        self.train_step_count = 0

        # Actor update interval
        self.actor_update_interval = 2

        # Store replay buffer
        self.store_replay_buffer = store_replay_buffer

        # Moving average calculator
        self.moving_aveg_calculator = MovingAvegCalculator(window_length = 1000)

        self.observe_separate_reward = True

    def normal_like(self, tensor, mean, std):
        return torch.normal(
            mean = torch.zeros_like(tensor).to(tensor.device) + mean,
            std = std
        )

    def normal(self, size, mean, std):
        return torch.normal(
            mean = torch.zeros(size, dtype = torch.float32) + mean,
            std = std
        ).to(self.device)

    def get_augmented_reward_dyna_based(self, state, action, next_state):
        state = torch.tensor(state, dtype = torch.float32).to(self.device)
        action = torch.tensor(action, dtype = torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype = torch.float32).to(self.device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            next_state = next_state.unsqueeze(0)

        reward = self.world_model.get_extra_reward(state, action, next_state) * \
                 max(self.moving_aveg_calculator.last_std, 0.1)

        return reward

    def train_step(self):
        state_batch, action_batch, reward1_batch, reward2_batch, \
            next_state_batch, done_batch = self.memory.sample(self.training_batch_size)

        # Add perturbation to next_action_batch
        next_action_batch = self.actor_target(next_state_batch) + \
                            self.normal_like(action_batch, mean = 0.0, std = self.epsilon).clamp(min = -0.5, max = 0.5)
        next_action_batch = next_action_batch.clamp(min = self.action_range[0], max = self.action_range[1])

        # World model update
        if self.recent_memory.full():
            recent_state_batch, recent_action_batch, _, _, recent_next_state_batch, _ = \
                self.recent_memory.sample(self.training_batch_size, no_done_sample=True)

            s = torch.cat((recent_state_batch, state_batch), dim=0)

            a = torch.cat((recent_action_batch, action_batch), dim=0)

            s_prim = torch.cat((recent_next_state_batch, next_state_batch), dim=0)

            self.world_model.train_step(s, a, s_prim)

        # Prepare critic target
        with torch.no_grad():
            # Q(s',pi(s'))
            # Q for reward1
            next_Q_values1 = self.critic1_target(
                next_state_batch,
                next_action_batch
            )

            next_Q_values2 = self.critic2_target(
                next_state_batch,
                next_action_batch
            )

            next_Q_values = torch.min(next_Q_values1, next_Q_values2)

            # target of Q(s,a)
            target_Q1_batch = reward1_batch.unsqueeze(-1) + self.gamma * \
                              (1.0 - done_batch.unsqueeze(-1)) * next_Q_values

            # Q for reward 2
            next_Q_values3 = self.critic3_target(
                next_state_batch,
                self.actor_target(next_state_batch)
            )

            next_Q_values4 = self.critic4_target(
                next_state_batch,
                self.actor_target(next_state_batch)
            )

            next_Q_values = torch.min(next_Q_values3, next_Q_values4)

            # target of Q(s,a)
            target_Q2_batch = reward2_batch.unsqueeze(-1) + self.gamma * \
                              (1.0 - done_batch.unsqueeze(-1)) * next_Q_values

        # Critic update
        self.critic1_optim.zero_grad()

        Q_batch = self.critic1(state_batch, action_batch)
        value_loss = self.mseLoss(Q_batch, target_Q1_batch)
        value_loss.backward()

        self.critic1_optim.step()

        self.critic2_optim.zero_grad()

        Q_batch = self.critic2(state_batch, action_batch)
        value_loss = self.mseLoss(Q_batch, target_Q1_batch)
        value_loss.backward()

        self.critic2_optim.step()

        self.critic3_optim.zero_grad()

        Q_batch = self.critic1(state_batch, action_batch)
        value_loss = self.mseLoss(Q_batch, target_Q2_batch)
        value_loss.backward()

        self.critic3_optim.step()

        self.critic4_optim.zero_grad()

        Q_batch = self.critic4(state_batch, action_batch)
        value_loss = self.mseLoss(Q_batch, target_Q2_batch)
        value_loss.backward()

        self.critic4_optim.step()

        if self.train_step_count % self.actor_update_interval == 0:
            # Actor update (Std)
            self.actor_optim_std.zero_grad()

            policy_loss = -self.critic1(
                state_batch,
                self.actor(state_batch, mode = "Std")
            ).mean()
            policy_loss.backward()

            self.actor_optim_std.step()

            # Actor update (Ent)
            enlarged_state_batch = state_batch.view(1, self.training_batch_size, -1).repeat(
                self.K, 1, 1).view(self.K * self.training_batch_size, -1)
            ent_action_batch = self.actor(enlarged_state_batch, mode = "Ent")

            with torch.no_grad():
                # Construct Gaussian kernel
                o_ij = ent_action_batch.view(self.K, 1, self.training_batch_size, -1) - \
                    ent_action_batch.view(1, self.K, self.training_batch_size, -1)
                K_ij = torch.exp(-o_ij.pow(2).sum(dim = 3) / (2 * math.pow(self.sigma, 2)))

                # Gradient of the Gaussian kernel
                K_ij_grad = K_ij.unsqueeze(-1) * o_ij / math.pow(self.sigma, 2)

            # Gradient from the Q function Q(s,a) to action a
            ent_action_batch_with_grad = ent_action_batch.detach().clone()
            ent_action_batch_with_grad.requires_grad = True

            Q_sa_ij = self.critic1(
                enlarged_state_batch,
                ent_action_batch_with_grad
            )
            Q_sa_ij = Q_sa_ij + self.critic3(
                enlarged_state_batch,
                ent_action_batch_with_grad
            )
            Q_sa_ij.sum().backward()

            dQ_da = ent_action_batch_with_grad.grad

            # Term 1 contains the generalized gradient
            term1 = (K_ij.view(self.K, self.K, self.training_batch_size, 1) *
                     dQ_da.view(1, self.K, self.training_batch_size, -1)).sum(dim = 1).view(
                self.K * self.training_batch_size, -1)

            # Term 2 refers to the maximum entropy term
            term2 = K_ij_grad.sum(dim = 1).view(self.K * self.training_batch_size, -1)

            # Update policy (Ent)
            self.actor_optim_ent.zero_grad()

            ent_action_batch.backward(-(term1 + self.balance_temperature * term2).detach() /
                                      self.K / self.training_batch_size / (self.balance_temperature + 1))

            self.actor_optim_ent.step()

            # Target network update
            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic1_target, self.critic1, self.tau)
            self.soft_update(self.critic2_target, self.critic2, self.tau)
            self.soft_update(self.critic3_target, self.critic3, self.tau)
            self.soft_update(self.critic4_target, self.critic4, self.tau)

        # Add train step count
        self.train_step_count += 1

    def action(self, state, mode = "train"):
        if mode == "train":
            action = self.to_numpy(
                self.actor(self.to_tensor(state).unsqueeze(0), mode = "Ent").squeeze(0) +
                self.normal([self.action_params["dims"]], mean = 0.0, std = self.sigma)
            )
            action = np.clip(action, self.action_range[0], self.action_range[1])
        elif mode == "test":
            action = self.to_numpy(
                self.actor_target(self.to_tensor(state).unsqueeze(0), mode = "Std").squeeze(0)
            )
        else:
            raise NotImplementedError()

        return action

    def train(self):
        self.actor_target.train()

    def eval(self):
        self.actor_target.eval()

    def reset(self):
        pass

    def observe(self, state, action, reward1, reward2, done):
        self.memory.append(state, action, reward1, reward2, done)
        self.recent_memory.append(state, action, reward1, reward2, done)

        self.moving_aveg_calculator.add_number(reward1)

    def to(self, device = None):
        if device is None:
            self.actor.to(self.device)
            self.actor_target.to(self.device)
            self.critic1.to(self.device)
            self.critic1_target.to(self.device)
            self.critic2.to(self.device)
            self.critic2_target.to(self.device)
            self.critic3.to(self.device)
            self.critic3_target.to(self.device)
            self.critic4.to(self.device)
            self.critic4_target.to(self.device)
            self.world_model.to(self.device)
        else:
            self.actor.to(device)
            self.actor_target.to(device)
            self.critic1.to(device)
            self.critic1_target.to(device)
            self.critic2.to(device)
            self.critic2_target.to(device)
            self.critic3.to(device)
            self.critic3_target.to(device)
            self.critic4.to(device)
            self.critic4_target.to(device)
            self.world_model.to(device)

    def load(self, folder_path):
        save_file_path = os.path.join(folder_path, "models.pt")

        if not os.path.exists(save_file_path):
            return False

        checkpoint = torch.load(save_file_path, map_location = 'cpu')
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.actor_optim_std.load_state_dict(checkpoint["actor_optim_std"])
        self.actor_optim_ent.load_state_dict(checkpoint["actor_optim_ent"])
        self.critic1.load_state_dict(checkpoint["critic"])
        self.critic1_target.load_state_dict(checkpoint["critic_target"])
        self.critic1_optim.load_state_dict(checkpoint["critic_optim"])
        self.critic2.load_state_dict(checkpoint["critic"])
        self.critic2_target.load_state_dict(checkpoint["critic_target"])
        self.critic2_optim.load_state_dict(checkpoint["critic_optim"])
        self.critic3.load_state_dict(checkpoint["critic"])
        self.critic3_target.load_state_dict(checkpoint["critic_target"])
        self.critic3_optim.load_state_dict(checkpoint["critic_optim"])
        self.critic4.load_state_dict(checkpoint["critic"])
        self.critic4_target.load_state_dict(checkpoint["critic_target"])
        self.critic4_optim.load_state_dict(checkpoint["critic_optim"])
        self.world_model.load_state_dict(checkpoint["world_model"])

        if self.store_replay_buffer:
            path = os.path.join(folder_path, "replay_mem.bin")
            with open(path, 'rb') as f:
                self.memory = pickle.load(f)

            path = os.path.join(folder_path, "recent_replay_mem.bin")
            with open(path, 'rb') as f:
                self.recent_memory = pickle.load(f)

        return True

    def save(self, folder_path):
        save_file_path = os.path.join(folder_path, "models.pt")

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        torch.save(
            {'actor': self.actor.state_dict(),
             'actor_target': self.actor_target.state_dict(),
             'actor_optim_std': self.actor_optim_std.state_dict(),
             'actor_optim_ent': self.actor_optim_ent.state_dict(),
             'critic1': self.critic1.state_dict(),
             'critic1_target': self.critic1_target.state_dict(),
             'critic1_optim': self.critic1_optim.state_dict(),
             'critic2': self.critic2.state_dict(),
             'critic2_target': self.critic2_target.state_dict(),
             'critic2_optim': self.critic2_optim.state_dict(),
             'critic3': self.critic3.state_dict(),
             'critic3_target': self.critic3_target.state_dict(),
             'critic3_optim': self.critic3_optim.state_dict(),
             'critic4': self.critic4.state_dict(),
             'critic4_target': self.critic4_target.state_dict(),
             'critic4_optim': self.critic4_optim.state_dict(),
             'world_model': self.world_model.state_dict()},
            save_file_path
        )

        if self.store_replay_buffer and np.random.random() < 0.1:
            path = os.path.join(folder_path, "replay_mem.bin")
            with open(path, 'wb') as f:
                pickle.dump(self.memory, f)

            path = os.path.join(folder_path, "recent_replay_mem.bin")
            with open(path, 'wb') as f:
                pickle.dump(self.recent_memory, f)
