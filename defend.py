import torch
import numpy as np
from utils import print_stat

class Gaussian:
    def __init__(self, eps):
        self.eps = eps
        self.generator = torch.Generator()

    def defense(self, x1):
        return torch.randn(x1.size(), generator=self.generator) * self.eps

    def print_info(self, train_acc, test_acc, attack_acc):
        print('gaussian noise with eps = ', self.eps)
        print_stat('train_acc', train_acc)
        print_stat('test_acc', test_acc)
        print_stat('attack_acc', attack_acc)
        print('')

class Defense:
    def __init__(self, d1, binary_features):
        self.d1 = d1
        self.binary_features = binary_features

    @staticmethod
    def print_info(train_acc, test_acc, attack_acc):
        print('misguided approach')
        print_stat('train_acc', train_acc)
        print_stat('test_acc', test_acc)
        print_stat('attack_acc', attack_acc)
        print('')
