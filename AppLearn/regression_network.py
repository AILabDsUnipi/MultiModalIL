import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Net(nn.Module):
	'''
	Simple NN
	'''
	def __init__(self, num_features, num_hidden, num_output):
		super(Net, self).__init__()
		self.num_features = num_features
		self.num_hidden = num_hidden
		self.num_output = num_output
		lr = 1e-2

		self.hidden = nn.Linear(self.num_features, self.num_hidden)
		self.relu = nn.ReLU()
		self.predict = nn.Linear(self.num_hidden, self.num_output)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()


	def forward(self, state):
		out = self.hidden(state)
		out = self.relu(out)
		out = self.predict(out)

		return out


	def train(self, num_episodes, x, y):
		x = T.from_numpy(x).float()
		y = T.from_numpy(y).float()
		X, Y = Variable(x), Variable(y)
		Y = Y.view(len(Y), 1)
		for i in range(num_episodes):
			prediction = self.forward(X)
			
			loss_ = self.loss(prediction, Y)
			self.optimizer.zero_grad()
			loss_.backward()
			self.optimizer.step()
			if i % 1000 == 0:
				print("Loss at episode " + str(i) + ": " +str(loss_.item()))