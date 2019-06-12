import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter

import gym
import numpy as np
from gym import wrappers
import env

import torch
from ddpg import DDPG
from normalized_actions import normalize,R_normalize,bullet_normalize
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
import random
import gym.spaces as spaces


np.random.seed(10)
torch.manual_seed(10)
INI = 0
gamma = 0.99
tau = 0.001
hidden_size = 128
observation_space = 4
replay_size = 10000
batch_size = 128
action_space = spaces.Box(np.array([0,0]),np.array([4,1]))

agentR = DDPG(gamma,tau,hidden_size,observation_space,action_space)
agentB = DDPG(gamma,tau,hidden_size,observation_space,action_space)
memoryR = ReplayMemory(replay_size)
memoryB = ReplayMemory(replay_size)

ounoise = OUNoise(action_space.shape[0])
rewardsR = []
rewardsB = []
total_numsteps = 0
updates = 0

writer = SummaryWriter()
Env = env.env(10)

for i_episode in range(1000):
    episode_reward_R = 0
    episode_reward_B = 0
    epi_reward_R = []
    epi_reward_B = []
    red_win = 0
    blue_win = 0
    for j in range(10):
        battle_reward_R = INI
        battle_reward_B = INI
        stateR,stateB,_,_,_,_ = Env.reset()
        bullet_R = stateR[2]
        bullet_B = stateR[3]
        stateR = torch.Tensor(stateB)
        stateB = torch.Tensor(stateB)
        while True and Env.done == False:
            dis = Env.calcdist(Env.R_Soldier[0], Env.B_Soldier[0])
            actionR = agentR.select_action(stateR,ounoise,None)
            actionB = agentB.select_action(stateB,ounoise,None)
            stateR_,stateB_,RR,RB,done,_ = Env.stepshoot(normalize(actionR,bullet_R),normalize(actionB,bullet_B))
            total_numsteps += 1
            battle_reward_R += RR
            battle_reward_B += RB

            actionR = torch.Tensor(bullet_normalize(actionR,bullet_R))
            actionB = torch.Tensor(bullet_normalize(actionB,bullet_B))
            mask = torch.Tensor([not done])
            next_state_R = torch.Tensor(stateR_)
            next_state_B = torch.Tensor(stateB_)
            rewardR = torch.Tensor([RR])
            rewardB = torch.Tensor([RB])

            memoryR.push(torch.Tensor(stateR),actionR,mask,next_state_R,rewardR)
            memoryB.push(torch.Tensor(stateB),actionB,mask,next_state_B,rewardB)

            stateR = torch.Tensor(stateR_)
            stateB = torch.Tensor(stateB_)

            if len(memoryR) > batch_size:
                for _ in range(5):
                    transitionsR = memoryR.sample(batch_size)
                    transitionsB = memoryB.sample(batch_size)
                    batchR = Transition(*zip(*transitionsR))
                    batchB = Transition(*zip(*transitionsB))
                    value_loss_R,policy_loss_R = agentR.update_parameters(batchR)
                    value_loss_B,policy_loss_B = agentB.update_parameters(batchB)

                    updates += 1
                if done:
                    break

        if Env.win == 1:
            red_win += 1
        elif Env.win == -1:
            blue_win += 1
        else:
            red_win += 0.5
            blue_win += 0.5
        epi_reward_R.append(battle_reward_R)
        epi_reward_B.append(battle_reward_B)
    episode_reward_R = np.mean(epi_reward_R)
    episode_reward_B = np.mean(epi_reward_B)
    rewardsR.append(episode_reward_R)
    rewardsB.append(episode_reward_B)

    print("Episode: {}, Red reward: {},Blue reward:{},red win: {},blue win: {}".format(i_episode, episode_reward_R,episode_reward_B,red_win,blue_win))



