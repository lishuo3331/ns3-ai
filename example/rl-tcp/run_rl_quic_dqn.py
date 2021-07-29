from py_interface import *
from ctypes import *
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--output_dir', type=str,
                    default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')

featureNum = 8
actionNum = 100000

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


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(featureNum, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, actionNum),
        )

    def forward(self, x):
        return self.layers(x)


class DQN(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()
        self.learn_step = 0
        self.batchsize = 32
        self.observer_shape = featureNum
        self.target_replace = 100
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((2000, 2*featureNum+2))    # s, a, r, s'
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=0.001)
            # self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() > 0.8 ** self.memory_counter:    # choose best
            action = self.eval_net.forward(x)
            action = torch.argmax(action, 0).numpy()
        else:    # explore
            action = np.random.randint(0, actionNum)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self, ):
        self.learn_step += 1
        if self.learn_step % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        sample = self.memory[sample_list, :]
        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(
            sample[:, self.observer_shape:self.observer_shape+1])
        r = torch.Tensor(
            sample[:, self.observer_shape+1:self.observer_shape+2])
        s_ = torch.Tensor(sample[:, self.observer_shape+2:])
        q_eval = self.eval_net(s).gather(1, a)
        q_next = self.target_net(s_).detach()
        q_target = r + 0.8 * q_next.max(1, True)[0].data

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

memorysize = 2048
#Uid:1234
Init(1234, memorysize)
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
    dqn = DQN()
exp = Experiment(1234, memorysize, 'rl-tcp', '../../')
exp.run(show_output=0)

try:
    while not var.isFinish():
        with var as data:
            if not data:
                break
        
            print(data.env.useRl)
            if data.env.useRl==0: 
                # print(data.env.useRl)
                continue
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
            useRl = data.env.useRl

            # print(rttGradient)
            # print(useRl)
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
                # if cWnd != data.act.newcWnd:
                s = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                a = dqn.choose_action(s)
                # 30305 后续替换为
                # new_cWnd = min(38305,max(a,10505))
                new_cWnd = a + 1460
                # if a & 1:
                #     new_cWnd = max(cWnd + segmentSize,1)
                # else:
                #     new_cWnd = max(int(cWnd / 2),1)
                    # new_cWnd = max(cWnd - int(max(1, (segmentSize * segmentSize) / max(cWnd,1))),segmentSize)
                if a < 3:
                    new_ssThresh = 2 * segmentSize
                else:
                    new_ssThresh = int(bytesInFlight / 2)
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
                # r = segmentsAcked - bytesInFlight - cWnd
                # print(pacingRate)
                # print(rtt)
                # print(lossRate)
                # r = 0.8 * math.log(max(pacingRate,1)) -  math.log(max(pacingRate,1)) * 500 * math.log(max(rtt,0.00001)) - 500 * lossRate
                # r = 0.8 * math.log(max(pacingRate,1))  -   math.log(max(pacingRate,1))  *  math.log(1 + max(rtt,0.00001)) -  0.00008 * lossRate
                r = 0.8 * pacingRate  -   pacingRate  * rttGradient   -  0.5 * lossRate
                
                print(a,cWnd,new_cWnd,r,rttGradient)

                # s_ = [ssThresh, cWnd, segmentsAcked,
                #       segmentSize, bytesInFlight]
                s_ = [ssThresh, cWnd, segmentsAcked,segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                dqn.store_transition(s, a, r, s_)

                if dqn.memory_counter > dqn.memory_capacity:
                    dqn.learn()
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
