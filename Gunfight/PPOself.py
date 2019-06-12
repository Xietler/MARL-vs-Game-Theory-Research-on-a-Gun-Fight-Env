import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import matplotlib
from utils import ou_reward_ppo,ou_ppo
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
import numpy as np
import argparse
import datetime
import math
import gym
import env
from scipy import stats
import sys


Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
EPS = 1e-10
RESULT_DIR = joindir('../result', '.'.join(__file__.split('.')[:-1]))
mkdir(RESULT_DIR, exist_ok=True)


# 超参数设置
class args(object):
    seed = 10
    num_episode = 200
    batch_size = 2048
    max_step_per_round = 1000
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 10
    minibatch_size = 256
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01
    lr = 3e-4
    num_parallel_run = 5
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True


# 运行状态
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


# Actor-Critic核心
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()

        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

        self.step = 0

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        #run policy network (actor) as well as value network (critic)
        #:param states: a Tensor2 represents states
        #:return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        #given mean and std, sample an action from normal(mean, std)
        #also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        #return probability of chosen the given actions under corresponding states of current network
        #:param states: Tensor
        #:param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba


# Importance Sampling
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


# PPO Core
def ppo(args):
    Env = env.env(10)
    # env = gym.make(args.env_name)
    # num_inputs = env.observation_space.shape[0]
    # num_actions = env.action_space.shape[0]
    num_inputs = 4
    num_actions = 10
    # env.seed(args.seed)
    torch.manual_seed(args.seed)

    networkR = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)
    networkB = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)
    optimizerR = opt.Adam(networkR.parameters(), lr=args.lr)
    optimizerB = opt.Adam(networkB.parameters(), lr=args.lr)

    running_state = ZFilter((num_inputs,), clip=5.0)

    # record average 1-round cumulative reward in every episode
    reward_record_R = []
    reward_record_B = []
    global_steps = 0

    lr_now = args.lr
    clip_now = args.clip

    for i_episode in range(args.num_episode):
        # 当前策略更新
        memoryR = Memory()
        memoryB = Memory()
        num_steps = 0
        reward_list_R = []
        reward_list_B = []
        len_list = []
        while num_steps < args.batch_size:
            stateR, stateB, _, _, _, _ = Env.reset()
            if args.state_norm:
                stateR = running_state(stateR)
                stateB = running_state(stateB)
            reward_sum_R = 0
            reward_sum_B = 0
            for t in range(args.max_step_per_round):
                action_mean_R, action_logstd_R, value_R = networkR(Tensor(stateR).unsqueeze(0))
                action_mean_B, action_logstd_B, value_B = networkR(Tensor(stateB).unsqueeze(0))
                actionR, logprobaR = networkR.select_action(action_mean_R, action_logstd_R)
                actionB, logprobaB = networkB.select_action(action_mean_B, action_logstd_B)
                actionR = actionR.data.numpy()[0]
                actionB = actionB.data.numpy()[0]
                actionR_real = np.argmax(actionR,axis = 0)
                actionB_real = np.argmax(actionB, axis=0)
                if Env.R_Soldier[0].bullet_count == 0:
                    actionR_real = actionR_real % 5
                if Env.B_Soldier[0].bullet_count == 0:
                    actionB_real = actionB_real % 5
                logprobaR = logprobaR.data.numpy()[0]
                logprobaB = logprobaB.data.numpy()[0]
                dis = Env.calcdist(Env.R_Soldier[0], Env.B_Soldier[0])
                #actionR = Env.R_Soldier[0].act(dis)
                next_state_R, next_state_B, reward_R, reward_B, done, _ = Env.stepshoot(actionR_real, actionB_real)
                reward_sum_R += reward_R
                reward_sum_B += reward_B
                if args.state_norm:
                    next_state_R = running_state(next_state_R)
                    next_state_B = running_state(next_state_B)
                mask = 0 if done else 1

                memoryR.push(stateR, value_R, actionR, logprobaR, mask, next_state_R, reward_R)
                memoryB.push(stateB, value_B, actionB, logprobaB, mask, next_state_B, reward_B)

                if done:
                    networkR.step += 1
                    networkB.step += 1
                    break

                stateR = next_state_R
                stateB = next_state_B

            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list_R.append(reward_sum_R)
            reward_list_B.append(reward_sum_B)
            len_list.append(t + 1)
        reward_record_R.append({
            'episode': i_episode,
            'steps': global_steps,
            'meanepreward': np.mean(reward_list_R),
            'meaneplen': np.mean(len_list)})

        reward_record_B.append({
            'episode': i_episode,
            'steps': global_steps,
            'meanepreward': np.mean(reward_list_B),
            'meaneplen': np.mean(len_list)})

        batchR = memoryR.sample()
        batchB = memoryB.sample()
        batch_size = len(memoryR)

        # Sample Parameters
        rewardsR = Tensor(batchR.reward)
        valuesR = Tensor(batchR.value)
        masksR = Tensor(batchR.mask)
        actionsR = Tensor(batchR.action)
        statesR = Tensor(batchR.state)
        oldlogprobaR = Tensor(batchR.logproba)

        rewardsB = Tensor(batchB.reward)
        valuesB = Tensor(batchB.value)
        masksB = Tensor(batchB.mask)
        actionsB = Tensor(batchB.action)
        statesB = Tensor(batchB.state)
        oldlogprobaB = Tensor(batchB.logproba)

        returnsR = Tensor(batch_size)
        deltasR = Tensor(batch_size)
        advantagesR = Tensor(batch_size)

        returnsB = Tensor(batch_size)
        deltasB = Tensor(batch_size)
        advantagesB = Tensor(batch_size)

        prev_return_R = 0
        prev_value_R = 0
        prev_advantage_R = 0

        prev_return_B = 0
        prev_value_B = 0
        prev_advantage_B = 0
        for i in reversed(range(batch_size)):
            returnsR[i] = rewardsR[i] + args.gamma * prev_return_R * masksR[i]
            deltasR[i] = rewardsR[i] + args.gamma * prev_value_R * masksR[i] - valuesR[i]
            advantagesR[i] = deltasR[i] + args.gamma * args.lamda * prev_advantage_R * masksR[i]

            returnsB[i] = rewardsB[i] + args.gamma * prev_return_B * masksB[i]
            deltasB[i] = rewardsB[i] + args.gamma * prev_value_B * masksB[i] - valuesB[i]
            advantagesB[i] = deltasB[i] + args.gamma * args.lamda * prev_advantage_B * masksB[i]

            prev_return_R = returnsR[i]
            prev_value_R = valuesR[i]
            prev_advantage_R = advantagesR[i]

            prev_return_B = returnsB[i]
            prev_value_B = valuesB[i]
            prev_advantage_B = advantagesB[i]


        if args.advantage_norm:
            advantagesR = (advantagesR - advantagesR.mean()) / (advantagesR.std() + EPS)
            advantagesB = (advantagesB - advantagesB.mean()) / (advantagesB.std() + EPS)

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            # current batch sample
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_statesR = statesR[minibatch_ind]
            minibatch_actionsR = actionsR[minibatch_ind]
            minibatch_oldlogprobaR = oldlogprobaR[minibatch_ind]
            #print(minibatch_statesR,minibatch_actionsR)
            minibatch_newlogprobaR = networkR.get_logproba(minibatch_statesR, minibatch_actionsR)
            minibatch_advantagesR = advantagesR[minibatch_ind]
            minibatch_returnsR = returnsR[minibatch_ind]
            minibatch_newvaluesR = networkR._forward_critic(minibatch_statesR).flatten()

            minibatch_statesB = statesB[minibatch_ind]
            minibatch_actionsB = actionsB[minibatch_ind]
            minibatch_oldlogprobaB = oldlogprobaB[minibatch_ind]
            minibatch_newlogprobaB = networkB.get_logproba(minibatch_statesB, minibatch_actionsB)
            minibatch_advantagesB = advantagesB[minibatch_ind]
            minibatch_returnsB = returnsB[minibatch_ind]
            minibatch_newvaluesB = networkB._forward_critic(minibatch_statesB).flatten()

            ratioR = torch.exp(minibatch_newlogprobaR - minibatch_oldlogprobaR)
            ratioB = torch.exp(minibatch_newlogprobaB - minibatch_oldlogprobaB)

            surr1R = ratioR * minibatch_advantagesR
            surr2R = ratioR.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantagesR
            loss_surrR = - torch.mean(torch.min(surr1R, surr2R))

            surr1B = ratioB * minibatch_advantagesB
            surr2B = ratioB.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantagesB
            loss_surrB = - torch.mean(torch.min(surr1B, surr2B))

            # value clip in the paper,not work
            if args.lossvalue_norm:
                minibatch_return_6stdR = 6 * minibatch_returnsR.std()
                loss_valueR = torch.mean((minibatch_newvaluesR - minibatch_returnsR).pow(2)) / minibatch_return_6stdR
                minibatch_return_6stdB = 6 * minibatch_returnsB.std()
                loss_valueB = torch.mean((minibatch_newvaluesB - minibatch_returnsB).pow(2)) / minibatch_return_6stdB
            else:
                loss_valueR = torch.mean((minibatch_newvaluesR - minibatch_returnsR).pow(2))
                loss_valueB = torch.mean((minibatch_newvaluesB - minibatch_returnsB).pow(2))

            loss_entropyR = torch.mean(torch.exp(minibatch_newlogprobaR) * minibatch_newlogprobaR)
            loss_entropyB = torch.mean(torch.exp(minibatch_oldlogprobaB) * minibatch_newlogprobaB)

            total_lossR = loss_surrR + args.loss_coeff_value * loss_valueR + args.loss_coeff_entropy * loss_entropyR
            total_lossB = loss_surrB + args.loss_coeff_value * loss_valueB + args.loss_coeff_entropy * loss_entropyB
            optimizerR.zero_grad()
            optimizerB.zero_grad()
            total_lossR.backward()
            total_lossB.backward()
            optimizerR.step()
            optimizerB.step()

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        # adam learning rate
        if args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            for g in optimizerR.param_groups:
                g['lr'] = lr_now
            for h in optimizerB.param_groups:
                h['lr'] = lr_now

        if i_episode % args.log_num_episode == 0:
            print('Episode: {} Red Reward: {:.4f} Red Loss = {:.4f} Blue Reward: {:.4f} Blue Loss = {:.4f}' \
                  .format(i_episode, ou_reward_ppo(reward_record_R[-1]['meanepreward'],networkR.step),total_lossR.data,ou_reward_ppo(reward_record_B[-1]['meanepreward'],networkB.step),total_lossB.data))
            print('-----------------')

    return networkB

def testppo(train,turn,battle,random_seed,nash_lower = 20,nash_upper = 20):
    a = args
    a.num_episode = train
    networkB = ppo(a)
    torch.manual_seed(10)
    np.random.seed(random_seed)
    Env = env.env(10)
    num_inputs = 4
    red = []
    blue = []
    for i_turn in range(turn):

        red_win = 0
        blue_win = 0
        running_state = ZFilter((num_inputs,), clip=5.0)

        for j in range(battle):
            stateR,stateB,RR,RB,done,_ = Env.reset()
            while Env.done == False:
                stateR_last = stateR
                stateB_last = stateB
                stateB = running_state(stateB)
                action_mean,action_logstd,value = networkB(Tensor(stateB).unsqueeze(0))
                action,logprob = networkB.select_action(action_mean,action_logstd)
                action = action.data.numpy()[0]
                actionB = ou_ppo(np.argmax(action,axis = 0),stateR_last,stateB_last,networkB.step)
                dis = Env.calcdist(Env.R_Soldier[0],Env.B_Soldier[0])
                actionR = Env.R_Soldier[0].act(dis)
                next_stateR, next_state, reward_R, reward, done, _ = Env.stepshoot(actionR, actionB)

                stateR = next_stateR
                stateB = next_state

                if done:
                    break

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

    print("Blue Win: {},T test: {}".format(np.mean(blue), stats.ttest_rel(red, blue)))



def test(argv):
    testppo(int(argv[0]),int(argv[1]),int(argv[2]),int(argv[3]),int(argv[4]),int(argv[5]))

if __name__ == '__main__':
    test(sys.argv[1:])



    
