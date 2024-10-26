#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lhys
# File  : demo.py

import torch
# 路径操作
import os
# 读取命令行参数
import sys
# 保存结果
import json
# 记录运行时间
import time
# 操作数据
import numpy as np
# 读取数据
import pandas as pd
# 复制
from copy import deepcopy
# 多进程
from multiprocessing import Pool, freeze_support
# 加载环境
from env import GraphEnvironment
# 加载模型
from models import DQN, DDQN, SARSA, QLearning

# 不显示警告
import warnings
warnings.filterwarnings("ignore")

# 路径
DATA_PATH = 'data'
RESULT_PATH = 'result'

make_dir_exist = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
make_dir_exist(DATA_PATH)
make_dir_exist(RESULT_PATH)

# 参数
episodes = 1000
_, start_node, done_node = sys.argv
start_node, done_node = map(int, [start_node, done_node])
start_gpu = 3 - start_node

hidden_size = 640
batch_size = 64
gamma = 0.6
lr = 0.1

probs = pd.read_excel(os.path.join(DATA_PATH, 'probs.xlsx'), usecols=[1]).to_numpy()

# 所有节点连接关系
files = ['节点对应关系（总）.xlsx', '两层之间连接（总）.xlsx']
total_edges = pd.concat(
    [pd.read_excel(os.path.join(DATA_PATH, file), usecols=range(2), header=0) for file in files]).to_numpy()

# 所有节点特征及节点与楼层对应关系
node_features = pd.read_excel(os.path.join(DATA_PATH, '节点特征.xlsx'), index_col=0).to_numpy()
node_features, total_layers = node_features[:, :-1], node_features[:, -1].astype(int)

# 根据最终节点选择总行动空间
# 选择最终节点楼层及之前楼层中所有节点
total_layer_num = total_layers[done_node]
state_size = np.where(total_layers <= total_layer_num)[0].size
index_list = np.where(total_edges < state_size)[0]
edges = total_edges[[index for index, count in zip(*np.unique(index_list, return_counts=True)) if count > 1]]
layers = total_layers[:state_size]
probs = probs[:state_size]

# 智能体训练函数
def train_per_agent(agent, start_node, keep_name, episodes=1000, flag=False):
    agent_name = keep_name.split('_')[0]
    # 计算时间
    start_time = time.time()
    result = {
        'best_path': [],
        'rewards': [],
        'loss': []
    }
    for episode in range(episodes):
        # 重置环境
        agent.env.reset()
        # 初始化
        state = start_node
        done = False
        path, rewards, loss = [], 0, []
        # 开始训练
        while not done:
            # 选择动作
            next_action = action = agent.choose_action(state)
            # 更新状态
            next_state, reward, done = agent.env.step(action)
            # 针对 SARSA 模型需要获得下个时间步的动作
            if flag: next_action = agent.choose_action(next_state)
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done, next_action)
            # 智能体学习
            agent_loss = agent.learn()
            # 输出结果
            action = next_action if flag else action
            print('\rEpisode [{:>4}/{:>4}] "{:>10}" state:{:3d}, action:{:3d}, reward:{:>8.3f}, done:{}'.format(
                episode + 1,
                episodes,
                agent_name,
                state,
                action,
                reward,
                done
            ), end='')
            # 更新状态
            state = next_state
            # 记录结果
            path.append(int(state))
            rewards += reward
            if agent_loss: loss.append(agent_loss)
        # 结束训练
        else:
            # 更新模型
            agent.update_target_model()
            # 记录结果
            result['rewards'].append(rewards)
            result['loss'].append(torch.tensor(loss).mean().item())
            # 三种情况保存最优路径
            # 1. 当前无最优路径
            # 2. 当前路径惩罚值大于过往最大惩罚值
            # 3. 当前路径惩罚值等于过往最大惩罚值 且 当前路径比最优路径短
            if not result['best_path'] or rewards > max(result['rewards']) \
                    or (rewards == max(result['rewards']) and len(path) < len(result['best_path'])):
                result['best_path'] = path
        # print(f'\rEpisode [{episode + 1}/{episodes}] total cost time {round(time.time() - start_time, 3)}s', end='')
        # 保存结果
        with open(os.path.join(RESULT_PATH, f'{keep_name}.json'), 'w') as f:
            json.dump(result, f)

    # 输出单个模型每个 episode 所需时间
    print(
        f'\nFinish agent {agent_name} No.{start_node} train, total cost time: {time.time() - start_time}s                             \n')


def init_agent(name, batch):
    env = GraphEnvironment(
        edges,
        layers,
        done_node,
        probs
    )
    agents = {
        'DQN': DQN(1, hidden_size, batch_size, lr, gamma, deepcopy(env), gpu=(start_gpu + 0 + batch) % 4),
        'DDQN': DDQN(1, hidden_size, batch_size, lr, gamma, deepcopy(env), gpu=(start_gpu + 0 + batch) % 4),
        'SARSA': SARSA(state_size, lr, gamma, deepcopy(env), gpu=(start_gpu + 0 + batch) % 4),
        'Q-Learning': QLearning(state_size, lr, gamma, deepcopy(env), gpu=(start_gpu + 0 + batch) % 4)
    }
    return agents[name], f'{name}_{start_node}_{done_node}'

if __name__ == '__main__':
    freeze_support()

    batch_list = [0]

    with Pool(processes=4) as pool:
        agent_list = [
            (*init_agent(agent, batch), agent == 'SARSA')
            for batch in batch_list
            for agent in [
                'DQN',
                'DDQN',
                'SARSA',
                'Q-Learning'
            ]
        ]
        pool.starmap(train_per_agent, [(agent, start_node, name, episodes, flag) for agent, name, flag in agent_list])

    pool.close()
    pool.join()