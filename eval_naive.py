import torch
import torch.nn as nn
import numpy as np
from utils import accuracy, powerset
from attack import equality_solve
from defend import Defense

def eval(net, data, bf):
    train_dataset, train_loader, test_dataset, test_loader = data
    criterion = nn.CrossEntropyLoss()
    train_acc = 0.0
    test_acc = 0.0
    A = []
    X = []
    D1 = net.d1
    if isinstance(net.defense, Defense):
        D1 = D1 - net.defense.nd + net.defense.nf
    # extract intermediate output
    def hook_forward_fn(module, input, output):
        A.append(output.numpy()[:, :D1])
    net.inter.register_forward_hook(hook_forward_fn)
    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader):
            X.append(data.numpy())
            output = net(data)
            loss = criterion(output, target)
            train_acc += accuracy(output, target).item() * len(data)
        train_acc /= len(train_dataset)
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            X.append(data.numpy())
            output = net(data)
            loss = criterion(output, target)
            test_acc += accuracy(output, target).item() * len(data)
        test_acc /= len(test_dataset)
    A = np.concatenate(A, axis=0)
    X = np.concatenate(X, axis=0)
    ans = equality_solve(A)
    print('total {} solution(s).'.format(len(ans)))
    idx, best_acc = 0, 0
    for i in range(net.d1):
        for sol in ans:
            acc = np.sum(np.isclose(X[:, i], sol))
            if acc == X.shape[0]:
                print('attack feature no.{} successfully.'.format(i))
            acc /= X.shape[0]
            if acc > best_acc:
                idx, best_acc = i, acc
    for feats in powerset(bf):
        nf = len(feats)
        if nf < 2:
            continue
        for sign in range(1 << nf):
            feat_sum = np.zeros(X.shape[0])
            for i in range(nf):
                feat_sum += X[:, feats[i]] * (1 if (sign & (1 << i)) == 0 else -1)
            for sol in ans:
                acc = np.sum(np.isclose(feat_sum, sol))
                if acc == X.shape[0]:
                    print('attack combination of features no.{} successfully.'.format(feats))
                acc /= X.shape[0]
                if acc > best_acc:
                    idx, best_acc = feats, acc
    '''for sol in ans:
        if np.sum(np.isclose(X[:, -1], sol)) == X.shape[0]:
            print('attack fake feature successfully.'.format(i))'''
    return train_acc, test_acc, best_acc, idx
