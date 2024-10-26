#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lhys
# File  : DDQN.py

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from .base import QNetwork, ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_dims, hidden_size, batch_size, lr, gamma, env, max_mem_size=100000, eps_end=0.01, eps_dec=0.01, gpu=0):
        super(DQN, self).__init__()
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.env = env
        # 输入维度，即总节点数
        self.input_dims = input_dims
        # 最大动作数
        self.action_size = max([len(env.graph[node]) for node in range(env.layers.size)])
        # 抽取样本数
        self.batch_size = batch_size
        # epsilon-greedy 策略参数
        self.epsilon = 1.0
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        # 折扣因子
        self.gamma = gamma
        # 策略网络
        self.model = QNetwork(input_dims, hidden_size, self.action_size).to(self.device)
        # 学习率
        self.lr = lr
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # 误差
        self.criterion = nn.MSELoss()
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(max_mem_size, input_dims, self.device)

    def choose_action(self, state):
        action_num = len(self.env.get_neighbors(state))
        if np.random.rand() < self.epsilon:
            return random.choice(range(action_num))
        else:
            state = torch.tensor([state], dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                self.model.eval()
                action_values = self.model(state)
            self.model.train()
            return torch.argmax(action_values[0, :action_num]).detach().cpu().item()

    def store_transition(self, state, action, reward, state_, done, *args):
        state, state_ = map(lambda x: torch.tensor([x], dtype=torch.float32), [state, state_])
        self.replay_buffer.store_transition(state, action, reward, state_, done)

    def learn(self, **kwargs):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_buffer(batch_size)
        # (batch_size, input_dims)
        states = states.float().to(self.device)
        # (batch_size, input_dims)
        next_states = next_states.float().to(self.device)
        # (batch_size, 1)
        actions = actions.long().to(self.device).unsqueeze(1)
        # (batch_size, 1)
        rewards = rewards.float().to(self.device).unsqueeze(1)
        # (batch_size, )
        dones = dones.to(self.device)

        current_q_values = self.model(states).gather(1, actions)

        target_q_values = rewards + self.gamma * current_q_values * dones.logical_not()

        loss = self.criterion(current_q_values, target_q_values)

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()

    def update_target_model(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

class DDQN(nn.Module):
    def __init__(self, input_dims, hidden_size, batch_size, lr, gamma, env, max_mem_size=100000, eps_end=0.01, eps_dec=0.01, gpu=0):
        super(DDQN, self).__init__()
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.env = env
        # 输入维度，即总节点数
        self.input_dims = input_dims
        # 最大动作数
        self.action_size = max([len(env.graph[node]) for node in range(env.layers.size)])
        # 抽取样本数
        self.batch_size = batch_size
        # epsilon-greedy 策略参数
        self.epsilon = 1.0
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        # 折扣因子
        self.gamma = gamma
        # 学习率
        self.lr = lr
        # 策略网络
        self.model = QNetwork(input_dims, hidden_size, self.action_size).to(self.device)
        # 目标网络
        self.target_model = QNetwork(input_dims, hidden_size, self.action_size).to(self.device)
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # 误差
        self.criterion = nn.MSELoss()
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(max_mem_size, input_dims, self.device)

    def choose_action(self, state):
        action_num = len(self.env.get_neighbors(state))
        if np.random.rand() < self.epsilon:
            return random.choice(range(action_num))
        else:
            state = torch.tensor([state], dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                self.model.eval()
                action_values = self.model(state)
            self.model.train()
            return torch.argmax(action_values[0, :action_num]).detach().cpu().item()

    def store_transition(self, state, action, reward, state_, done, *args):
        state, state_ = map(lambda x: torch.tensor([x], dtype=torch.float32), [state, state_])
        self.replay_buffer.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_buffer(self.batch_size)
        # (batch_size, input_dims)
        states = states.float().to(self.device)
        # (batch_size, input_dims)
        next_states = next_states.float().to(self.device)
        # (batch_size, 1)
        actions = actions.long().to(self.device).unsqueeze(1)
        # (batch_size, 1)
        rewards = rewards.float().to(self.device).unsqueeze(1)
        # (batch_size, )
        dones = dones.bool().to(self.device)

        Q_targets_next = self.target_model(next_states).detach().max(dim=1)[0].unsqueeze(1)
        Q_targets_next[dones] = 0.0
        Q_targets = rewards + self.gamma * Q_targets_next * dones.logical_not()
        Q_expected = self.model(states).gather(1, actions)

        loss = self.criterion(Q_expected, Q_targets)
        self.soft_update()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()

    def soft_update(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.lr / 10 * local_param.data + (1.0 - self.lr / 10) * target_param.data)

    def update_target_model(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        self.target_model.load_state_dict(self.model.state_dict())