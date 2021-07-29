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


curr_path = str(Path().absolute())
parent_path = str(Path().absolute().parent)
sys.path.append(parent_path) # add current terminal path to sys.path

parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--output_dir', type=str,
                    default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')

featureNum = 8
actionNum = 4
actionDim = 100000

class TcpRlEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ('nodeId', c_uint32),
        ('socketUid', c_uint32),
        ('envType', c_uint8),
        ('simTime_us', c_int64),
        ('ssThresh', c_uint32), # maxcwnd
        ('cWnd', c_uint32), 
        ('segmentSize', c_uint32), # 最大分段大小
        ('segmentsAcked', c_uint32), # 接收到的段数量
        ('bytesInFlight', c_uint32), # 已发送 未ack的包
        ('pacingRate', c_uint64), # 吞吐量
        ('rtt', c_double), # 吞吐量
        ('lossRate', c_uint64), # 丢包率乘以pacingRate
        ('rttGradient', c_double), # 延时
        
    ]


class TcpRlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('new_ssThresh', c_uint32),
        ('new_cWnd', c_uint32)
    ]


class QlearningConfig:
    '''训练相关参数'''
    def __init__(self):
        self.train_eps = 200 # 训练的episode数目
        self.gamma = 0.9 # reward的衰减率
        self.epsilon_start = 0.99 # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 200 # e-greedy策略中epsilon的衰减率
        self.lr = 0.1 # learning rate        

class QLearning(object):
    def __init__(self,action_dim,cfg):
        self.action_dim = action_dim  # dimension of acgtion
        self.lr = cfg.lr  # learning rate
        self.gamma = cfg.gamma  
        self.epsilon = 0.7 
        self.sample_count = 0  
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table  = defaultdict(lambda: np.zeros(action_dim)) # A nested dictionary that maps state -> (action -> action-value)

    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        # e-greedy policy
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.action_dim) 
        return action
            
    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward  # terminal state
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)]) 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
        
    def save(self,path):
        import dill
        torch.save(
            obj=self.Q_table,
            f=path+"Qleaning_model.pkl",
            pickle_module=dill
        )
    

    def load(self, path):
        import dill
        self.Q_table =torch.load(f=path+'Qleaning_model.pkl',pickle_module=dill)

    def learn(self,cfg):
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        sample = self.memory[sample_list, :]

        state = torch.Tensor(sample[:, :self.observer_shape])
        action = torch.LongTensor(
            sample[:, self.observer_shape:self.observer_shape+1])
        reward = torch.Tensor(
            sample[:, self.observer_shape+1:self.observer_shape+2])
        next_state = torch.Tensor(sample[:, self.observer_shape+2:])

        action = self.choose_action(state)  # 根据算法选择一个动作
        next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
        self.update(state, action, reward, next_state, done)  # Q-learning算法更新
        state = next_state  # 存储上一个观察值
memorysize = 2048
Init(1234, memorysize) # Init(shmKey, memSize)
var = Ns3AIRL(1234, TcpRlEnv, TcpRlAct)
res_list = ['ssThresh_l', 'cWnd_l', 'segmentsAcked_l',
            'segmentSize_l', 'bytesInFlight_l','pacingRate_l','rtt_l','lossRate_l']
args = parser.parse_args()

if args.result:
    for res in res_list:
        globals()[res] = []
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

if args.use_rl:
    cfg = QlearningConfig()
    # 动作维度 2
    qlearn = QLearning(actionDim,cfg)
    # dqn = DQN()
exp = Experiment(1234, memorysize, 'rl-tcp', '../../')
exp.run(show_output=0)
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
                s = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                # a = dqn.choose_action(s)
                #动作选择
                a = qlearn.choose_action(s)
                print(a)
                new_cWnd = a
                new_ssThresh = ssThresh
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
                print(pacingRate)
                print(rtt)
                # r = 0.8 * math.log(pacingRate) - math.log(pacingRate) * 500 * math.log(rtt)
                r = 0.8 * pacingRate  -   pacingRate  * rttGradient   -  0.5 * lossRate

                # r = - 500 * math.log(rtt)

                # s_ = [ssThresh, cWnd, segmentsAcked,
                #       segmentSize, bytesInFlight,pacingRate,rtt]
                s_ = [ssThresh, cWnd, segmentsAcked,segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                # Q table 更新
                #def update(self, state, action, reward, next_state, done):
                qlearn.update(s,a,r,s_,1)
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
