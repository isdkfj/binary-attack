from utils import get_args, set_random_seed
from data import load_data
from train import prepare_dataset, train
from eval import eval
from defend import Gaussian, Defense
from module import Net

args = get_args()
set_random_seed(args.seed)

# last column of X is fabricated label
train_X, test_X, train_Y, test_Y = load_data(args.data, args.path, args.seed)
print(train_X[0, :])
print(test_X[0, :])

if args.data == 'bank':
    num_classes = 2
    dimensions = [8]
    hid = [60, 30, 10]
    binary_features = [7]
elif args.data =='credit':
    num_classes = 2
    dimensions = [10]
    hid = [50, 20, 10]
    binary_features = [1]
elif args.data == 'mushroom':
    num_classes = 2
    dimensions = [15]
    hid = [50, 20]
    binary_features = [3]
    # swap binary features out
    train_X[:, [5, 6, 7, 9, 10, 11, 12, 13]] = train_X[:, [10, 11, 12, 13, 5, 6, 7, 9]]
    test_X[:, [5, 6, 7, 9, 10, 11, 12, 13]] = test_X[:, [10, 11, 12, 13, 5, 6, 7, 9]]
elif args.data == 'nursery':
    num_classes = 5
    dimensions = [6]
    hid = [200, 100, 50]
    binary_features = [5]
elif args.data == 'covertype':
    num_classes = 7
    dimensions = [11]
    hid = args.net
    binary_features = [10]

train_dataset, train_loader, test_dataset, test_loader = prepare_dataset(train_X, train_Y, test_X, test_Y, args.bs)

def run_exp(d1, num_exp, mask):
    list_train_acc = []
    list_test_acc = []
    list_attack_acc = []
    for iter_exp in range(num_exp):
        net = Net(d1, train_X.shape[1] - d1 - 1, num_classes, hid, mask.defense)
        train(net, (train_dataset, train_loader), verbose=args.verbose)
        train_acc, test_acc, attack_acc, idx = eval(net, (train_dataset, train_loader, test_dataset, test_loader), binary_features)
        list_train_acc.append(train_acc)
        list_test_acc.append(test_acc)
        list_attack_acc.append(attack_acc)
        print(train_acc, test_acc, attack_acc)
    mask.print_info(list_train_acc, list_test_acc, list_attack_acc)

for d1 in dimensions:
    '''gauss = Gaussian(0.0)
    run_exp(d1, args.repeat, gauss)
    gauss = Gaussian(0.01)
    run_exp(d1, args.repeat, gauss)
    gauss = Gaussian(0.05)
    run_exp(d1, args.repeat, gauss)
    gauss = Gaussian(0.1)
    run_exp(d1, args.repeat, gauss)
    gauss = Gaussian(0.25)
    run_exp(d1, args.repeat, gauss)
    gauss = Gaussian(0.5)
    run_exp(d1, args.repeat, gauss)'''
    fab = Defense(d1)
    run_exp(d1, args.repeat, fab)
        
        
