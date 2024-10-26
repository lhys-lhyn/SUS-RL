#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lhys
# File  : Sarsa.py

import torch
import torch.nn as nn

import random

class SARSA(nn.Module):
    def __init__(self, num_states, learning_rate, gamma, env, eps_end=0.01, eps_dec=0.001, gpu=0):
        super(SARSA, self).__init__()
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.env = env
        self.Q = torch.zeros((num_states, max([len(env.graph[node]) for node in range(env.layers.size)])), dtype=torch.float32).to(self.device)
        self.learning_rate = learning_rate
        self.gamma = gamma
        # epsilon-greedy 策略参数
        self.epsilon = 1.0
        self.eps_min = eps_end
        self.eps_dec = eps_dec

    def choose_action(self, state):
        action_num = len(self.env.get_neighbors(state))
        if np.random.rand() < self.epsilon:
            return random.choice(range(action_num))
        else:
            return torch.argmax(self.Q[state, :action_num]).detach().cpu().item()

    def store_transition(self, state, action, reward, next_state, done, next_action):
        self.current = [state, action, reward, next_action, next_state, done]

    def learn(self):
        state, action, reward, next_action, next_state, done = self.current
        predict = self.Q[state, action]
        target = reward + self.gamma * self.Q[next_state, next_action]
        loss = target - predict
        self.Q[state, action] += self.learning_rate * loss
        return loss.mean().mean()

    def update_target_model(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
