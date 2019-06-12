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
import sys

import torch
from ddpg import DDPG
from normalized_actions import normalize,R_normalize,bullet_normalize
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import OptReplayMemory, Transition
import random
import gym.spaces as spaces
from utils import ou_norm,ou_reward
from scipy import stats

def OptDDPG_TrainSelf(episode,battle):
    np.random.seed(10)
    torch.manual_seed(10)
    gamma = 0.99
    tau = 0.001
    hidden_size = 128
    observation_space = 4#observation space:self pos,enemy pos,self bullet,enemy bullet
    replay_size = 10000
    batch_size = 128
    action_space = spaces.Box(np.array([0,0]),np.array([4,1]))#predict a space [0,4] and [0,1] to represent the action
    threshold = 2 #replay threshold
    agentR = DDPG(gamma,tau,hidden_size,observation_space,action_space)
    agentB = DDPG(gamma,tau,hidden_size,observation_space,action_space)
    memoryR = OptReplayMemory(replay_size)
    memoryB = OptReplayMemory(replay_size)

    ounoise = OUNoise(action_space.shape[0]) #add a ounoise to enhance the data
    rewardsR = []
    rewardsB = []
    total_numsteps = 0
    updates = 0

    writer = SummaryWriter()
    Env = env.env(10)#init

    for i_episode in range(episode):
        episode_reward_R = 0
        episode_reward_B = 0
        epi_reward_R = []
        epi_reward_B = []
        batch_R = []
        batch_B = []
        red_win = 0
        blue_win = 0
        for j in range(battle):
            battle_reward_R = 0
            battle_reward_B = 0
            stateR,stateB,_,_,_,_ = Env.reset()#reset the env
            bullet_R = stateR[2]
            bullet_B = stateR[3]
            state_judge_R = sum(stateR)
            state_judge_B = sum(stateB)
            stateR = torch.Tensor(stateB)
            stateB = torch.Tensor(stateB)
            while True and Env.done == False:
                dis = Env.calcdist(Env.R_Soldier[0], Env.B_Soldier[0])
                #no param noise
                actionR = agentR.select_action(stateR,ounoise,None)
                actionB = agentB.select_action(stateB,ounoise,None)
                #update the normalize action
                stateR_,stateB_,RR,RB,done,_ = Env.stepshoot(normalize(actionR,bullet_R),normalize(actionB,bullet_B))
                state_judge_R += normalize(actionR,bullet_R)
                state_judge_B += normalize(actionB,bullet_B)
                #add the state to the batch
                batch_R.append(state_judge_B)
                batch_B.append(state_judge_R)
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

                if RB - RR > threshold:
                    memoryR.pushP(torch.Tensor(stateB),actionB,mask,next_state_B,rewardB)
                    memoryR.pushL(torch.Tensor(stateR),actionR,mask,next_state_R,rewardR)
                    memoryR.pushLR(state_judge_R)
                    memoryB.pushP(torch.Tensor(stateB),actionB,mask,next_state_B,rewardB)
                    memoryB.pushL(torch.Tensor(stateR),actionR,mask,next_state_R,rewardR)
                    memoryB.pushLR(state_judge_R)
                elif RR - RB > threshold:
                    memoryR.pushP(torch.Tensor(stateR),actionR,mask,next_state_R,rewardR)
                    memoryR.pushL(torch.Tensor(stateB),actionB,mask,next_state_B,rewardB)
                    memoryR.pushLR(state_judge_B)
                    memoryB.pushP(torch.Tensor(stateR),actionR,mask,next_state_R,rewardR)
                    memoryB.pushL(torch.Tensor(stateB),actionB,mask,next_state_B,rewardB)
                    memoryB.pushLR(state_judge_R)
                else:
                    if random.random() < 0.2:
                        memoryR.pushP(torch.Tensor(stateR),actionR,mask,next_state_R,rewardR)
                        memoryR.pushL(torch.Tensor(stateB),actionB,mask,next_state_B,rewardB)
                        memoryB.pushP(torch.Tensor(stateR), actionR, mask, next_state_R, rewardR)
                        memoryB.pushL(torch.Tensor(stateB), actionB, mask, next_state_B, rewardB)

                if done:
                    break

                stateR = torch.Tensor(stateR_)
                stateB = torch.Tensor(stateB_)
            #Experience Replay according to the priority
            if len(memoryR.profit_memory) > batch_size and len(memoryR.loss_memory) > batch_size and i_episode % 10 == 5:
                memoryR.judge(batch_R)
                memoryB.judge(batch_B)
                for _ in range(1):
                    transitionsR = memoryR.sample(batch_size)
                    transitionsB = memoryB.sample(batch_size)
                    batchR = Transition(*zip(*transitionsR))
                    batchB = Transition(*zip(*transitionsB))
                    value_loss_R,policy_loss_R = agentR.update_parameters(batchR)
                    value_loss_B,policy_loss_B = agentB.update_parameters(batchB)

                    updates += 1
                batch_R = []
                batch_B = []
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
            agentR.actor.step += 1
            agentB.actor.step += 1
            agentR.critic.step += 1
            agentB.critic.step += 1
        episode_reward_R = ou_reward(np.mean(epi_reward_R),gamma,agentR.actor.step)
        episode_reward_B = ou_reward(np.mean(epi_reward_B),gamma,agentB.actor.step)
        rewardsR.append(episode_reward_R)
        rewardsB.append(episode_reward_B)

        print("Episode: {}, Red reward: {},Blue reward:{},red win: {},blue win: {}".format(i_episode, episode_reward_R,
                                                                                           episode_reward_B, red_win,
                                                                                   blue_win))
    agentR.save_model("SelfR",str(episode*battle))
    agentB.save_model("SelfB",str(episode*battle))

def OptDDPGtestmodelNash(turn,battle,actor,critic,random_seed,nash_lower = 20,nash_upper = 20):
    np.random.seed(random_seed)
    torch.manual_seed(10)
    gamma = 0.99
    tau = 0.001
    hidden_size = 128
    observation_space = 4
    replay_size = 10000
    batch_size = 128
    threshold = 2
    action_space = spaces.Box(np.array([0, 0]), np.array([4, 1]))

    agent = DDPG(gamma, tau, hidden_size, observation_space, action_space)
    agent.load_model(actor,critic)
    ounoise = OUNoise(action_space.shape[0])
    rewards = []
    total_numsteps = 0
    updates = 0
    red = []
    blue = []

    Env = env.env(10)
    for i_turn in range(turn):
        episode_reward = 0
        epi_reward = []
        red_win = 0
        blue_win = 0
        for j in range(battle):
            battle_reward = 0
            stateR, stateB, _, _, _, _ = Env.reset()
            bullet_R = stateR[2]
            bullet_B = stateR[3]
            stateB = torch.Tensor(stateB)
            while True and Env.done == False:

                dis = Env.calcdist(Env.R_Soldier[0], Env.B_Soldier[0])
                #get the Nash Policy
                actionR = Env.R_Soldier[0].act(dis,nash_lower,nash_upper)
                action = agent.select_action(stateB, ounoise, None)
                actionB = ou_norm(normalize(action, bullet_B),stateR,stateB,agent.actor.step)

                stateR_, stateB_, RR, RB, done, _ = Env.stepshoot(actionR,actionB)
                total_numsteps += 1


                #update the state
                stateR = torch.Tensor(stateR_)
                stateB = torch.Tensor(stateB_)


            if Env.win == 1:
                red_win += 1
            elif Env.win == -1:
                blue_win += 1
            else:
                red_win += 0.5
                blue_win += 0.5
        red.append(red_win)
        blue.append(blue_win)

        print("Turn: {} ,red win: {},blue win: {}".format(i_turn, red_win, blue_win))
        print("-----------------")
    print("Blue Win: {},T test: {}".format(np.mean(blue),stats.ttest_rel(red,blue)))


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == 'train':
        OptDDPG_TrainSelf(int(args[1]),int(args[2]))
    elif args[0] == 'exec':
        OptDDPGtestmodelNash(int(args[1]),int(args[2]),args[3],args[4],int(args[5]),int(args[6]),int(args[7]))
    else:
        print("invalid command")