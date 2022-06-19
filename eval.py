import torch
import torch.nn as nn
from utils import accuracy

def eval(net, data):
    train_dataset, train_loader, test_dataset, test_loader = data
    criterion = nn.CrossEntropyLoss()
    train_acc = 0.0
    test_acc = 0.0
    A = []
    X = []
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
        test_acc = 0.0
        for i, (data, target) in enumerate(test_loader):
            X.append(data.numpy())
            output = net(data)
            loss = criterion(output, target)
            test_acc += accuracy(output, target).item() * len(data)
        test_acc /= len(test_dataset)
    A = np.concatenate(A, axis=0)
    X = np.concatenate(X, axis=0)
    sol, val = leverage_score_solve(A, 10, net.d1 + 1)
    rec = np.dot(A, sol.reshape(A.shape[1], 1))
    idx, best_acc = 0, 0
    for i in range(net.d1):
        acc = np.sum(np.isclose(X[:, i].reshape(-1, 1), rec > 0.5)) / X.shape[0]
        if acc > best_acc:
            idx, best_acc = i, acc
    return train_acc, test_acc, best_acc, idx
