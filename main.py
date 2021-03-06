from utils import get_args, set_random_seed
from data import load_data
from train import prepare_dataset, train
from defend import Gaussian, Defense
from module import Net
import numpy as np

args = get_args()
set_random_seed(args.seed)

if args.am == 'linear':
    from eval_naive import eval
else:
    from eval import eval

# last column of X is fabricated label
train_X, test_X, train_Y, test_Y = load_data(args.data, args.path, args.seed)

if args.data == 'bank':
    num_classes = 2
    d1 = 8
    hid = [60, 30, 10]
elif args.data =='credit':
    num_classes = 2
    d1 = 10
    hid = [100, 50, 20]
elif args.data == 'mushroom':
    num_classes = 2
    d1 = 15
    hid = [300, 200, 100]
    # swap binary features out
    #train_X[:, [5, 6, 7, 9, 15, 16, 17, 18]] = train_X[:, [15, 16, 17, 18, 5, 6, 7, 9]]
    #test_X[:, [5, 6, 7, 9, 15, 16, 17, 18]] = test_X[:, [15, 16, 17, 18, 5, 6, 7, 9]]
elif args.data == 'nursery':
    num_classes = 5
    d1 = 6
    hid = [200, 100]
elif args.data == 'covertype':
    num_classes = 7
    d1 = 10
    hid = [200, 200, 200]
    train_X[:, [6, 7, 8, 9, 10, 11, 12, 13]] = train_X[:, [10, 11, 12, 13, 6, 7, 8, 9]]
    test_X[:, [6, 7, 8, 9, 10, 11, 12, 13]] = test_X[:, [10, 11, 12, 13, 6, 7, 8, 9]]

binary_features = []
for i in range (d1):
    if np.sum(np.isclose(train_X[:, i], 0)) + np.sum(np.isclose(train_X[:, i], 1)) == train_X.shape[0] and np.sum(np.isclose(test_X[:, i], 0)) + np.sum(np.isclose(test_X[:, i], 1)) == test_X.shape[0]:
        binary_features.append(i)

print('binary features:', binary_features)

train_dataset, train_loader, validation_dataset, validation_loader, test_dataset, test_loader = prepare_dataset(train_X, train_Y, test_X, test_Y, args.bs)

def run_exp(d1, num_exp, defense):
    list_train_acc = []
    list_test_acc = []
    list_attack_acc = []
    for iter_exp in range(num_exp):
        net = Net(d1, train_X.shape[1] - d1 - 1, num_classes, hid, defense)
        train(net, (train_dataset, train_loader, validation_dataset, validation_loader), verbose=args.verbose)
        train_acc, test_acc, attack_acc, idx = eval(net, (validation_dataset, validation_loader, test_dataset, test_loader), binary_features)
        list_train_acc.append(train_acc)
        list_test_acc.append(test_acc)
        list_attack_acc.append(attack_acc)
        print(train_acc, test_acc, attack_acc, idx)
    defense.print_info(list_train_acc, list_test_acc, list_attack_acc)

if args.dm == 'gauss':
    gauss = Gaussian(args.eps)
    run_exp(d1, args.repeat, gauss)
elif args.dm == 'fake':
    fab = Defense(d1, binary_features)
    run_exp(d1, args.repeat, fab)
