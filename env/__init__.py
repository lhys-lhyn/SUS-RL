#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lhys
# File  : __init__.py

import torch
import numpy as np
import networkx as nx

class GraphEnvironment:
    def __init__(self, edges, layers, done, probs):
        self.layers = layers
        self.done = done
        self.graph = self.generate_3d_graph(edges)
        self.probs = probs
        self.steps = 0

    def generate_3d_graph(self, edges):
        '''
        创建 3D 拓扑结构作为 `环境`
        '''
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    def get_neighbors(self, node):
        '''
        获得当前节点的相邻节点
        '''
        return list(self.graph[node])

    def reset(self):
        '''
        使用随机起始节点初始化 `环境`
        '''
        self.state = torch.zeros(self.layers.size)
        self.steps = 0
        self.current = -1

    def step(self, action):
        '''
        根据动作计算奖励值
        '''
        # 通过索引选择节点
        action = self.get_neighbors(self.current)[action] if self.current >= 0 else action
        # 判断是否达到终止条件（到达终止节点为终止条件）
        if action == self.done: return action, 1000, 1
        reward = -0.1 * (np.max(self.layers) - self.layers[action]) * self.probs[action, 0] - (10 if self.state[action] else 0)
        self.steps += 1
        # 更新当前状态，选择的节点状态被标记为 1
        self.state[action] = 1
        self.current = action
        return action, reward, 0