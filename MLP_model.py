from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, num_class):
        super(MLP, self).__init__()
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(5120,2560)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        #self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(2560, num_class)
    def forward(self, x):
        x = self.flat(x)
        #x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
