import torch
import numpy as np
from utils import print_stat

class Gaussian:
    def __init__(self, eps):
        self.eps = eps
        self.train_generator = torch.Generator()
        self.val_generator = torch.Generator()
        self.generator = self.train_generator

    def noise(self, x1):
        return torch.randn(x1.size(), generator=self.generator) * self.eps

    def set_mode(self, mode):
        if mode == 'train':
            self.generator = self.train_generator
        else:
            self.generator = self.val_generator
            
    def print_info(self, train_acc, test_acc, attack_acc):
        print('gaussian noise with eps = ', self.eps)
        print_stat('train_acc', train_acc)
        print_stat('test_acc', test_acc)
        print_stat('attack_acc', attack_acc)
        print('')

class Defense:
    def __init__(self, d1, binary_features, nf=1, nd=1):
        self.d1 = d1
        self.binary_features = binary_features
        self.nf = nf
        self.nd = nd

    def set_mode(self, mode):
        pass

    @staticmethod
    def print_info(train_acc, test_acc, attack_acc):
        print('misguided approach')
        print_stat('train_acc', train_acc)
        print_stat('test_acc', test_acc)
        print_stat('attack_acc', attack_acc)
        print('')
