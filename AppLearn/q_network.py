import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
	'''
	Simple DQN
	'''
	def __init__(self, num_actions, mem_size, batch_size, alpha, gamma, num_features):
		super(Net, self).__init__()
		self.num_actions = num_actions
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.gamma = gamma
		self.alpha = alpha
		self.num_nodes = 100
		self.num_features = num_features

		self.fc_layer1 = nn.Linear(self.num_features, self.num_nodes)
		self.relu = nn.ReLU()
		self.fc_layer2 = nn.Linear(self.num_nodes, self.num_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		state = state.view(-1, self.num_features)
		out = self.fc_layer1(state)
		out = self.relu(out)
		out = self.fc_layer2(out)

		return out

	def adjust_lr(self):
		alpha_decay = 0.99
		self.lr = self.lr * alpha_decay
		
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr