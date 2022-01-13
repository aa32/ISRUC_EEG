import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):

	def __init__(self):
		super(SimpleRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, dropout=0.5, batch_first=True)
		self.fc = nn.Linear(in_features=32, out_features=5)

	def forward(self, x):
		x, _ = self.rnn(x)
		x = self.fc(x)
		return x[:, -1, :]


class RCNN(nn.Module):

	def __init__(self):
		super(RCNN, self).__init__()
		# RCNN - based on Biswal et al
		# CNN layer has 1 convolutional layer.
		# Averaged raw wave form in a single (1,3000) array input

		# 	conv1: 	20 1d filters of length 200
		# 	pool1: 	max pool size 20 stride 10
		# 	stack 	20 1d signals into 2d stack
		# 	conv2: 	400 filters size (20,30)
		# 	pool2: 	max pool size 10 stride 2
		# 	fc1: 	500 units
		# 	fc2: 	500 units
		# self.conv1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=200, stride=1, padding=1)
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 4497, out_features=64)
		self.fc2 = nn.Linear(64, 1)
		self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True, dropout=0.5)
		self.fc3 = nn.Linear(in_features=32, out_features=5)

	def forward(self, x):
		# print(x.shape)
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# print(x.shape)
		x = x.view(-1, 16 * 4497)
		# x = x.view(-1, 5120)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		# print(x.shape)
		# x = np.reshape(x, [32,32,1])
		# print(x.shape[0])
		if(x.shape[0]==192):
			x = x.view(32, 6, 1)
		else:
			x = x.view(x.shape[0],1,1)
		# print(x.shape)
		x, _ = self.rnn(x)
		# print(x.shape)
		x = self.fc3(x)
		return x[:, -1, :]
