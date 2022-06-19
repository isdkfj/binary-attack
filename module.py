import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, d1, d2, c, hidden, defense):
        super(Net, self).__init__()
        # d1 for passive party, d2 for active party
        self.d1 = d1
        self.d2 = d2
        self.input1 = nn.Linear(d1, hidden[0], bias=False)
        self.input2 = nn.Linear(d2, hidden[0], bias=True)
        self.hidden_layer = []
        for i in range(len(hidden) - 1):
            self.hidden_layer.append(nn.Linear(hidden[i], hidden[i + 1]))
        self.final = nn.Linear(hidden[-1], c)
        self.inter = nn.Identity()

    def forward(self, x):
        x1 = self.input1(x[:, :self.d1])
        x1 = x1 + defense(x1, x, self.input1.weight.detach())
        x1 = self.inter(x1)
        x2 = self.input2(x[:, self.d1: self.d1 + self.d2])
        x = x1 + x2
        x = F.relu(x)
        for layer in self.hidden_layer:
            x = F.relu(layer(x))
        x = self.final(x)
        return x
