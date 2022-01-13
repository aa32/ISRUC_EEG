import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 4497, out_features=64)
		self.fc2 = nn.Linear(64, 5)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 4497)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
		