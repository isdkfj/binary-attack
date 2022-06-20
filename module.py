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
        hidden_layers = []
        for i in range(len(hidden) - 1):
            hidden_layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)
        self.final = nn.Linear(hidden[-1], c)
        self.inter = nn.Identity()
        self.defense = defense

    def forward(self, x):
        x1 = self.input1(x[:, :self.d1])
        r = self.defense(x1, x, self.input1.weight.detach())
        ratio = float(torch.sum(x1 ** 2).numpy()) / float(torch.sum(r ** 2).numpy())
        print(ratio)
        #x1 = x1 + self.defense(x1, x, self.input1.weight.detach())
        x1 = self.inter(x1)
        x2 = self.input2(x[:, self.d1: self.d1 + self.d2])
        x = x1 + x2
        x = F.relu(x)
        x = self.hidden(x)
        x = self.final(x)
        return x
