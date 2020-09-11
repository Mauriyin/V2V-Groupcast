# Defination of Deep Q-Network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

# Initial Environment
class channel_env:
	def __init__(self,n_rbs):
		self.n_rbs = n_rbs
		self.reward = 1

		self.action_set = np.arange(n_rbs)
		self.action = -1
		self.observation = -1

# QNetwork
class ResNet(nn.Module):
    def __init__(self, state_size, action):
        super(ResNet, self).__init__()
        self.h1 = nn.Linear(state_size, 64)
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 64)
        self.h4 = nn.Linear(64, 64)
        self.h5 = nn.Linear(64, 64)
        self.h6 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action)

    def forward(self, x):
        h1 = F.relu(self.h1(x))
        h2 = F.relu(self.h2(h1))

        h3 = F.relu(self.h3(h2))
        h4 = F.relu(self.h4(h3)) + h2

        h5 = F.relu(self.h5(h4))
        h6 = F.relu(self.h6(h5)) + h4

        return self.out(h6)

class DQN(nn.Module):
	def __init__(self,
                 state_size,
                 action_size,
                 learning_rate=0.01,
				 epsilon=0.5,
                 epsilon_min=0.001,
                 epsilon_decay=0.5):
		super(DQN, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate
		self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

	def choose_action(self, state, action_index):
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
		action_size_period = len(action_index)
        if np.random.random() < self.epsilon:
			action_sel = np.random.choice(action_size_period)
            return action_index[action_sel]

		state = Variable(torch.from_numpy(state.astype(float))).float()
		if torch.cuda.is_available()
        	state = state.cuda()

        q_out = self.model(state)
		action = action_index[0]
		for i in range(1, action_size_period):
            if q_out[0][action_index[i]] > q_out[0][action]:
                action = action_index[i]

        return action


