import torch
import torch.nn as nn
import numpy as np
from utils import accuracy, powerset
from attack import leverage_score_solve

def eval(net, data, bf):
    train_dataset, train_loader, test_dataset, test_loader = data
    criterion = nn.CrossEntropyLoss()
    train_acc = 0.0
    test_acc = 0.0
    A = []
    X = []
    net.defense.set_mode('train')
    # extract intermediate output
    def hook_forward_fn(module, input, output):
        A.append(output.numpy()[:, :net.d1])
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
    sol, val = leverage_score_solve(A, 20, net.d1 + 1)
    cov = np.dot(A.T, A)
    for bid in bf:
        real_x = np.linalg.solve(cov, np.dot(A.T, X[:, bid].reshape(-1, 1)))
        print('error of feature no.{}:'.format(bid), np.sum((X[:, bid].reshape(-1, 1) - np.dot(A, real_x.reshape(A.shape[1], 1))) ** 2))
    print('error of solution:', val)
    rec = np.dot(A, sol.reshape(A.shape[1], 1))
    print('attack acc w.r.t. fake label:', np.sum(np.isclose(X[:, -1].reshape(-1, 1), rec > 0.5)) / X.shape[0])
    idx, best_acc = 0, 0
    for i in range(net.d1):
        acc = np.sum(np.isclose(X[:, i].reshape(-1, 1), rec > 0.5)) / X.shape[0]
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
            acc = np.sum(np.isclose(feat_sum.reshape(-1, 1), rec > 0.5)) / X.shape[0]
            if acc > best_acc:
                idx, best_acc = feats, acc
    return train_acc, test_acc, best_acc, idx
