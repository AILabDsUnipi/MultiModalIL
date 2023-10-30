import torch.nn as nn
import torch
from utils.math import *
import torch.nn.functional as F

class Policy(nn.Module):
	def __init__(self, state_dim, action_dim, code_dim, hidden_size=(100,100), activation='tanh', logstd=0):
		super().__init__()

		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid
		elif activation == 'leaky_relu':
			self.activation = F.leaky_relu		

		self.layer1 = nn.Linear(state_dim, hidden_size[0])
		self.layer2 = nn.Linear(hidden_size[0], hidden_size[1])

		self.c_layer = nn.Linear(code_dim, 32)

		self.hc_layer = nn.Linear(132, 32)

		self.action_mean = nn.Linear(32, action_dim)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * logstd)

		self.initialize_weights()

	def initialize_weights(self):
		nn.init.xavier_uniform_(self.layer1.weight)
		nn.init.constant_(self.layer1.bias, 0.0)

		nn.init.xavier_uniform_(self.layer2.weight)
		nn.init.constant_(self.layer2.bias, 0.0)

		nn.init.xavier_uniform_(self.c_layer.weight)
		nn.init.constant_(self.c_layer.bias, 0.0)

		nn.init.xavier_uniform_(self.hc_layer.weight)
		nn.init.constant_(self.hc_layer.bias, 0.0)

		nn.init.xavier_uniform_(self.action_mean.weight)
		nn.init.constant_(self.action_mean.bias, 0.0)


	def forward(self, x, c, bcloning=False):
		x = self.activation(self.layer1(x))
		x = self.activation(self.layer2(x))
		c = self.activation(self.c_layer(c))
		
		if bcloning:
			axis = 2
		else:
			axis = 1

		hc = torch.cat([x, c], axis=axis)

		hc = self.activation(self.hc_layer(hc))

		action_mean = self.action_mean(hc)
		action_log_std = self.action_log_std.expand_as(action_mean)

		action_std = torch.exp(action_log_std)

		return action_mean, action_log_std, action_std

	def select_action(self, x, c, episode):
		action_mean, _, action_std = self.forward(x, c, False)

		action = torch.normal(action_mean, action_std)

		return action

	def select_bc_action(self, x, c):
		action_mean, _, _ = self.forward(x, c, True)

		return action_mean

	def get_log_prob(self, x, c, actions):
		action_mean, action_log_std, action_std = self.forward(x, c)
		
		return normal_log_density(actions, action_mean, action_log_std, action_std)

class Value(nn.Module):
	def __init__(self, state_dim, hidden_size=(100,100), activation='tanh'):
		super().__init__()

		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid
		elif activation == 'leaky_relu':
			self.activation = F.leaky_relu

		self.affine_layers = nn.Sequential(
			nn.Linear(state_dim, hidden_size[0]),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_size[0], hidden_size[1]),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_size[1], 1)
		)

		self.initialize_weights()

	def initialize_weights(self):
		for m in self.affine_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0.0)

	def forward(self, x):
		value = self.affine_layers(x)

		return value

class Discriminator(nn.Module):
	def __init__(self, state_dim, action_dim, code_dim, hidden_size=(100,100), activation='tanh'):
		super().__init__()

		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid
		elif activation == 'leaky_relu':
			self.activation = F.leaky_relu

		self.affine_layers = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_size[0]),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_size[0], hidden_size[1]),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_size[1], 1),
			nn.Sigmoid()
		)	

		self.initialize_weights()

	def initialize_weights(self):
		for m in self.affine_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0.0)

	def forward(self, x):
		prob = self.affine_layers(x)

		return prob  

class Posterior(nn.Module):
	def __init__(self, num_inputs, code_dim, hidden_size=(100,100), activation='tanh'):
		super().__init__()

		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid
		elif activation == 'leaky_relu':
			self.activation = F.leaky_relu

		self.affine_layers = nn.Sequential(
			nn.Linear(num_inputs, hidden_size[0]),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_size[0], hidden_size[1]),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_size[1], code_dim)
		)

	def initialize_weights(self):
		for m in self.affine_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0.0)

	def forward(self, x):
		prob = self.affine_layers(x)

		return prob