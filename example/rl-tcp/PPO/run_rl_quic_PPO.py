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
from PPO import PPO
import glob
import time
import datetime
# import roboschool

# import pybullet_envs
import matplotlib.pyplot as plt
from PPO import PPO
# curr_path = os.path.dirname(__file__)
# parent_path=os.path.dirname(curr_path) 
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path) # add current terminal path to sys.path

from common.plot import plot_rewards
from common.utils import save_results
# from env import OUNoise
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
actionMax = 8*16*1024

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
        # ('newSimultionTime',c_double),
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


class PPOConfig:
    def __init__(self):
        ####### initialize environment hyperparameters ######

        # env_name = "RoboschoolWalker2d-v1"
        self.algo = 'PPO'
        self.env_name = "Pendulum-v0"
        # env_name = "CartPole-v1"
        self.has_continuous_action_space = True  # continuous action space; else discrete

        self.max_ep_len = 100                   # max timesteps in one episode
        self.max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

        self.print_freq = self.max_ep_len * 10        # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2           # log avg reward in the interval (in num timesteps)
        self.save_model_freq = int(1e5)          # save model frequency (in num timesteps)

        self.action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

        #####################################################


        ## Note : print/log frequencies should be > than max_ep_len


        ################ PPO hyperparameters ################

        self.update_timestep = self.max_ep_len * 4      # update policy every n timesteps
        self.K_epochs = 80               # update policy for K epochs in one PPO update

        self.eps_clip = 0.2          # clip parameter for PPO
        self.gamma = 0.99            # discount factor

        self.lr_actor = 0.0003       # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network

        self.random_seed = 0         # set random seed if required (0 = no random seed)

        #####################################################

memorysize = 1024
Init(1234, memorysize) # Init(shmKey, memSize)
var = Ns3AIRL(1234, TcpRlEnv, TcpRlAct)
res_list = ['ssThresh_l', 'cWnd_l', 'segmentsAcked_l',
            'segmentSize_l', 'bytesInFlight_l','pacingRate_l','rtt_l','lossRate_l']
args = parser.parse_args()
# ou_noise = OUNoise(actionDim,-1,1)
if args.result:
    for res in res_list:
        globals()[res] = []
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

if args.use_rl:
    # cfg = DDPGConfig()
    # 动作维度 2
    # agent = DDPG(stateDim,actionDim,cfg)
    # dqn = DQN()
    cfg = PPOConfig()
    agent = PPO(stateDim, actionDim, cfg.lr_actor, cfg.lr_critic, cfg.gamma, cfg.K_epochs, cfg.eps_clip, cfg.has_continuous_action_space, cfg.action_std)
exp = Experiment(1234, memorysize, 'rl-tcp', '../../')
exp.run(show_output=0)
reward_list = []
SimultionTime = 0
# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0
reward_list = []
reward_sum = 0

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

            # newSimultionTime = data.env.newSimultionTime
            # if newSimultionTime == SimultionTime:
                # continue
            # SimultionTime = newSimultionTime
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
                # action = agent.choose_action(s)
                action = agent.select_action(s)
                # aciton = np.clip(action,-1,1)
                # action = get_action(action)
                # action = ou_noise.get_action(action)[0]
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
                # if done 交互完成后，获取网络新状态
                if(cWnd == new_cWnd):
                    done = True
                else:
                    done = False
                ssThresh = data.env.ssThresh
                cWnd = data.env.cWnd
                segmentsAcked = data.env.segmentsAcked
                segmentSize = data.env.segmentSize
                bytesInFlight = data.env.bytesInFlight
                pacingRate = data.env.pacingRate
                rtt = data.env.rtt
                lossRate = data.env.lossRate
                rttGradient = data.env.rttGradient
                # newSimultionTime = data.env.newSimultionTime

                # modify the reward
                # r = 0.7 * segmentsAcked - 1.2 * bytesInFlight - 0.5 * cWnd  
                # r = 0.8 * math.log(pacingRate) - math.log(pacingRate) * 500 * math.log(rtt)
                r = 0.8 * pacingRate/(1024*1024)  -   pacingRate  * rttGradient/(1024*1024)   -  0.5 * lossRate/(1024*1024)
                
                if time_step % cfg.max_ep_len == 0:
                    reward_list.append(reward_sum)
                    print("eposide: ", time_step , "reward: ", reward_sum)
                    reward_sum = 0
                reward_sum += r
                time_step += 1
                # r = - 500 * math.log(rtt)

                # s_ = [ssThresh, cWnd, segmentsAcked,
                #       segmentSize, bytesInFlight,pacingRate,rtt]
                # s_ = [ssThresh, cWnd, segmentsAcked,segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                s_ = [bytesInFlight/(1024*1024),pacingRate/(1024*1024),rtt,lossRate/(1024*1024),rttGradient]
                # Q table 更新
                #def update(self, state, action, reward, next_state, done):
                # print("s",s)
                # print("s_",s_)
                # saving reward and is_terminals
                agent.buffer.rewards.append(r)
                agent.buffer.is_terminals.append(done)
                # agent.memory.push(s,[action],r,s_,done)
                if time_step % cfg.update_timestep == 0:
                    agent.update()
                if cfg.has_continuous_action_space and time_step % cfg.action_std_decay_freq == 0:
                    agent.decay_action_std(cfg.action_std_decay_rate, cfg.min_action_std)
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
    agent.save(checkpoint_path=SAVED_MODEL_PATH)
    plt.clf()
    plot_rewards(reward_list,tag="train",algo = cfg.algo,path=RESULT_PATH)
    save_results(reward_list,tag='train',path=RESULT_PATH)

