from py_interface import *
from ctypes import *
import os
import torch
import argparse
import utils
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import sys
from pathlib import Path
import gym
import copy
import torch.nn.functional as F


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
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

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
        ('newTime', c_double), # 仿真时间
    ]


class TcpRlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('new_ssThresh', c_uint32),
        ('new_cWnd', c_uint32)
    ]




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

#模型存放
if not os.path.exists("./models/td3"):
    os.makedirs("./models/td3")

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": args.discount,
    "tau": args.tau,
}
kwargs["policy_noise"] = args.policy_noise * max_action
kwargs["noise_clip"] = args.noise_clip * max_action
kwargs["policy_freq"] = args.policy_freq
policy = TD3(**kwargs)
#经验回放
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
episode_reward = 0
episode_timesteps = 0
episode_num = 0

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
    # 动作维度 2
    policy = TD3(**kwargs)
    # dqn = DQN()
exp = Experiment(1234, memorysize, 'rl-tcp', '../../')
exp.run(show_output=0)
# global newTime1
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
            newTime = data.env.newTime;
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
                episode_timesteps += 1
                # 环境状态
                # s = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]
                s = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight,pacingRate,rtt,lossRate]
                # a = dqn.choose_action(s)
                #动作选择
                a = (policy.select_action(np.array(s))).clip(-max_action, max_action)
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
                newTime1 = data.env.newTime
                if(newTime1 == newTime):
                    continue
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
                # qlearn.update(s,a,r,s_,1)
                # dqn.store_transition(s, a, r, s_)
                replay_buffer.add(s, a, s_, r, 1)

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

