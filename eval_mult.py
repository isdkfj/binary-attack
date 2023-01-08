import torch
import torch.nn as nn
import numpy as np
from utils import accuracy, powerset
from attack import equality_solve, leverage_score_solve
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
    sol = np.zeros(X.shape[0])
    ind = []
    for bits in range(1 << len(ans)):
        values = []
        for i in range(len(ans)):
            values.append((bits >> i) & 1)
        indices = np.ones(X.shape[0]).astype('bool')
        for i in range(len(ans)):
            indices = indices * (ans[i] == values[i])
        partial_A = A[indices, :]
        partial_A = partial_A - partial_A[0, :]
        partial_sol, partial_val = leverage_score_solve(partial_A, 20, D1 - net.defense.nf + 1)
        rec = np.dot(partial_A, partial_sol.reshape(partial_A.shape[1], 1))
        sol[indices] = (rec > 0.5).reshape(-1)
        ind.append(indices)
    idx, best_acc = 0, 0
    for sign in range(1 << len(ind)):
        for i in range(len(ind)):
            if ((sign >> i) & 1) == 1:
                sol[ind[i]] = 1 - sol[ind[i]]
        for i in range(net.d1):
            acc = np.sum(np.isclose(X[:, i], sol))
            if acc == X.shape[0]:
                print('attack feature no.{} successfully.'.format(i))
            acc /= X.shape[0]
            if acc > best_acc:
                idx, best_acc = i, acc
        for i in range(len(ind)):
            if ((sign >> i) & 1) == 1:
                sol[ind[i]] = 1 - sol[ind[i]]
    return train_acc, test_acc, best_acc, 
