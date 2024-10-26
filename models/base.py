#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lhys
# File  : base.py

# 导入所需的库
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, total_nodes, hidden_size, nodes_per_layer):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(total_nodes, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, nodes_per_layer)

    def forward(self, layer):
        x = torch.relu(self.fc1(layer))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer(nn.Module):
    def __init__(self, max_size, input_shape, device):
        super(ReplayBuffer, self).__init__()
        self.to(device)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, input_shape), dtype=torch.float32).to(device)
        self.new_state_memory = torch.zeros((self.mem_size, input_shape), dtype=torch.float32).to(device)
        self.action_memory = torch.zeros(self.mem_size, dtype=torch.int64).to(device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32).to(device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=bool).to(device)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones