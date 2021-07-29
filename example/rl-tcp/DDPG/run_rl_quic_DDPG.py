from py_interface import *
from ctypes import *
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import sys
from pathlib import Path
import gym
from agent import DDPG
import datetime
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

from common.plot import plot_rewards
from common.utils import save_results
from env import OUNoise
DDPG_path = os.getcwd()
# print(os.getcwd())
SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
SAVED_MODEL_PATH = DDPG_path+"/saved_model/"+SEQUENCE+'/' # path to save model
# print(parent_path+"/")
if not os.path.exists(DDPG_path+"/saved_model/"): os.mkdir(DDPG_path+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = DDPG_path+"/results/"+SEQUENCE+'/' # path to save rewards
if not os.path.exists(DDPG_path+"/results/"): os.mkdir(DDPG_path+"/results/")
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--output_dir', type=str,
                    default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')


stateDim = 5
# actionNum = 4
actionDim = 1

actionMin = 1460
actionMax = 16*1024

class TcpRlEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ('useRl',c_uint32), # 是否使用RL
        ('nodeId', c_uint32),
        ('socketUid', c_uint32),
        ('envType', c_uint8),
        # ('simTime_us', c_int64),
        ('ssThresh', c_uint32),
        ('cWnd', c_uint32),
        ('segmentSize', c_uint32),
        ('segmentsAcked', c_uint32),
        ('bytesInFlight', c_uint32),
        ('pacingRate', c_uint64), # 吞吐量
        ('rtt', c_double), # 延时
        ('lossRate', c_uint64), # 丢包率乘以pacingRate
        ('rttGradient', c_double), # 延时
        # ('use123',c_double)
    ]


class TcpRlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('new_ssThresh', c_uint32),
        ('new_cWnd', c_uint32)
    ]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_action(action):
    return np.clip(action,-1,1)

class DDPGConfig:
    def __init__(self):
        self.env = 'Pendulum-v0'
        self.algo = 'DDPG'
        self.gamma = 0.99
        self.critic_lr = 1e-3  
        self.actor_lr = 1e-4 
        self.memory_capacity = 1000000
        self.batch_size = 128
        self.train_eps =300
        self.eval_eps = 200
        self.eval_steps = 200
        self.target_update = 4
        self.hidden_dim = 30
        self.soft_tau=1e-2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(cfg,env,agent):
        print('Start to train ! ')
        # ou_noise = OUNoise(env.action_space) # action noise
        rewards = []
        ma_rewards = [] # moving average rewards
        ep_steps = []
        for i_episode in range(cfg.train_eps):
            state = env.reset()
            # ou_noise.reset()
            done = False
            ep_reward = 0
            i_step = 0
            while not done:
                i_step += 1
                action = agent.choose_action(state)
                # action = ou_noise.get_action(action, i_step)  # 即paper中的random process
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                agent.memory.push(state, action, reward, next_state, done)
                agent.update()
                state = next_state
            print('Episode:{}/{}, Reward:{}'.format(i_episode+1,cfg.train_eps,ep_reward))
            ep_steps.append(i_step)
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
            else:
                ma_rewards.append(ep_reward)
        print('Complete training！')
        return rewards,ma_rewards

memorysize = 2048
Init(1234, memorysize) # Init(shmKey, memSize)
var = Ns3AIRL(1234, TcpRlEnv, TcpRlAct)
res_list = ['ssThresh_l', 'cWnd_l', 'segmentsAcked_l',
            'segmentSize_l', 'bytesInFlight_l','pacingRate_l','rtt_l','lossRate_l']
args = parser.parse_args()
ou_noise = OUNoise(actionDim,-1,1)
if args.result:
    for res in res_list:
        globals()[res] = []
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

if args.use_rl:
    cfg = DDPGConfig()
    # 动作维度 2
    agent = DDPG(stateDim,actionDim,cfg)
    # dqn = DQN()
exp = Experiment(1234, memorysize, 'rl-tcp', '../../')
exp.run(show_output=0)
reward_list = []
try:
    while not var.isFinish():
        with var as data:
            if not data:
                break
    #         print(var.GetVersion())
            ssThresh = data.env.ssThresh
            cWnd = data.env.cWnd
            segmentsAcked = data.env.segmentsAcked
            segmentSize = data.env.segmentSize
            bytesInFlight = data.env.bytesInFlight
            pacingRate = data.env.pacingRate
            rtt = data.env.rtt
            lossRate = data.env.lossRate
            rttGradient = data.env.rttGradient
    #         print(ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight)

            if args.result:
                for res in res_list:
                    globals()[res].append(globals()[res[:-2]])
                    #print(globals()[res][-1])

            if not args.use_rl:
                new_cWnd = 1
                new_ssThresh = 1
                # IncreaseWindow
                if (cWnd < ssThresh):
                    # slow start
                    if (segmentsAcked >= 1):
                        new_cWnd = cWnd + segmentSize
                if (cWnd >= ssThresh):
                    # congestion avoidance
                    if (segmentsAcked > 0):
                        adder = 1.0 * (segmentSize * segmentSize) / max(cWnd,1)
                        adder = int(max(1.0, adder))
                        new_cWnd = cWnd + adder
                # GetSsThresh
                new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))
                data.act.new_cWnd = new_cWnd
                data.act.new_ssThresh = new_ssThresh
            else:
                # 环境状态
                # s = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]
                # s = [ssThresh, cWnd, segmentsAcked,segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                s = [bytesInFlight/(1024*1024),pacingRate/(1024*1024),rtt,lossRate/(1024*1024),rttGradient]
                # a = dqn.choose_action(s)
                #动作选择
                action = agent.choose_action(s)
                # action = get_action(action)
                action = ou_noise.get_action(action)[0]
                print("action",action)
                new_cWnd = int((action + 1.0) * float(actionMax - actionMin) / 2.0)
                new_ssThresh = ssThresh
                #
                #动作更新
                #
                # if a & 1:
                #     new_cWnd = cWnd + segmentSize
                # else:
                #     new_cWnd = cWnd + \
                #         int(max(1, (segmentSize * segmentSize) / max(cWnd,1)))
                # if a < 3:
                #     new_ssThresh = 2 * segmentSize
                # else:
                #     new_ssThresh = int(bytesInFlight / 2)
                data.act.new_cWnd = new_cWnd
                data.act.new_ssThresh = new_ssThresh
                ssThresh = data.env.ssThresh
                cWnd = data.env.cWnd
                segmentsAcked = data.env.segmentsAcked
                segmentSize = data.env.segmentSize
                bytesInFlight = data.env.bytesInFlight
                pacingRate = data.env.pacingRate
                rtt = data.env.rtt
                lossRate = data.env.lossRate
                rttGradient = data.env.rttGradient
                

                # modify the reward
                # r = 0.7 * segmentsAcked - 1.2 * bytesInFlight - 0.5 * cWnd  
                # r = 0.8 * math.log(pacingRate) - math.log(pacingRate) * 500 * math.log(rtt)
                r = 0.8 * pacingRate/(1024*1024)  -   pacingRate  * rttGradient/(1024*1024)   -  0.5 * lossRate/(1024*1024)
                reward_list.append(r)
                # r = - 500 * math.log(rtt)

                # s_ = [ssThresh, cWnd, segmentsAcked,
                #       segmentSize, bytesInFlight,pacingRate,rtt]
                # s_ = [ssThresh, cWnd, segmentsAcked,segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                s_ = [bytesInFlight/(1024*1024),pacingRate/(1024*1024),rtt,lossRate/(1024*1024),rttGradient]
                # Q table 更新
                #def update(self, state, action, reward, next_state, done):
                print("s",s)
                print("s_",s_)
                
                agent.memory.push(s,[action],r,s_,1)
                agent.update()
                # dqn.store_transition(s, a, r, s_)
                
                # if dqn.memory_counter > dqn.memory_capacity:
                    # dqn.learn()
except KeyboardInterrupt:
    exp.kill()
    del exp

if args.result:
    for res in res_list:
        y = globals()[res]
        x = range(len(y))
        plt.clf()
        plt.plot(x, y, label=res[:-2], linewidth=1, color='r')
        plt.xlabel('Step Number')
        plt.title('Information of {}'.format(res[:-2]))
        plt.savefig('{}.png'.format(os.path.join(args.output_dir, res[:-2])))
    agent.save(path=SAVED_MODEL_PATH)
    plt.clf()
    plot_rewards(reward_list,tag="train",algo = cfg.algo,path=RESULT_PATH)
    save_results(reward_list,tag='train',path=RESULT_PATH)

