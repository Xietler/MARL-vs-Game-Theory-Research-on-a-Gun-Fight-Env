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
from replay_memory import OptReplayMemory, Transition
import random
import gym.spaces as spaces


np.random.seed(10)
torch.manual_seed(10)
gamma = 0.99
tau = 0.001
hidden_size = 128
observation_space = 4
replay_size = 10000
batch_size = 128
threshold = 2
action_space = spaces.Box(np.array([0,0]),np.array([4,1]))

agent = DDPG(gamma,tau,hidden_size,observation_space,action_space)
memory = OptReplayMemory(replay_size)

ounoise = OUNoise(action_space.shape[0])
rewards = []
total_numsteps = 0
updates = 0

writer = SummaryWriter()
Env = env.env(10)

for i_episode in range(1000):
    episode_reward = 0
    epi_reward = []
    red_win = 0
    blue_win = 0
    for j in range(10):
        battle_reward = 0
        stateR,stateB,_,_,_,_ = Env.reset()
        bullet_R = stateR[2]
        bullet_B = stateR[3]
        stateB = torch.Tensor(stateB)
        while True and Env.done == False:

            dis = Env.calcdist(Env.R_Soldier[0], Env.B_Soldier[0])
            actionR = Env.R_Soldier[0].act(dis)
            action = agent.select_action(stateB,ounoise,None)
            stateR_,stateB_,RR,RB,done,_ = Env.stepshoot(actionR,normalize(action,bullet_B))
            total_numsteps += 1
            battle_reward += RB

            action = torch.Tensor(bullet_normalize(action,bullet_B))
            mask = torch.Tensor([not done])
            next_state = torch.Tensor(stateB_)
            next_state_R = torch.Tensor(stateR_)
            reward = torch.Tensor([RB])
            reward_R = torch.Tensor([RR])
            if RB - RR > threshold:
                memory.pushP(torch.Tensor(stateB),action,mask,next_state,reward)
                memory.pushL(torch.Tensor(stateR),R_normalize(actionR),mask,next_state_R,reward_R)
            elif RR - RB > threshold:
                memory.pushP(torch.Tensor(stateR),action,mask,next_state_R,reward_R)
                memory.pushL(torch.Tensor(stateB),R_normalize(actionR),mask,next_state,reward)


            stateR = torch.Tensor(stateR_)
            stateB = torch.Tensor(stateB_)

            if len(memory) > batch_size:
                for _ in range(5):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    value_loss,policy_loss = agent.update_parameters(batch)
                    writer.add_scalar('loss/value', value_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)

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
        writer.add_scalar('reward/train', battle_reward, 100*i_episode+j)
        epi_reward.append(battle_reward)
    episode_reward = np.mean(epi_reward)
    rewards.append(episode_reward)

    print("Episode: {}, reward: {},red win: {},blue win: {}".format(i_episode, episode_reward, red_win, blue_win))