from utils import get_args, set_random_seed
from data import load_data
from train import prepare_dataset, train
from eval_naive import eval
from defend import Gaussian
from module import Net

args = get_args()
set_random_seed(args.seed)

# last column of X is fabricated label
train_X, test_X, train_Y, test_Y = load_data(args.data, args.path, args.seed)
train_dataset, train_loader, test_dataset, test_loader = prepare_dataset(train_X, train_Y, test_X, test_Y, args.bs)

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
    hid = [50, 20]
    # swap binary features out
    train_X[:, [5, 6, 7, 9, 15, 16, 17, 18]] = train_X[:, [15, 16, 17, 18, 5, 6, 7, 9]]
    test_X[:, [5, 6, 7, 9, 15, 16, 17, 18]] = test_X[:, [15, 16, 17, 18, 5, 6, 7, 9]]
elif args.data == 'nursery':
    num_classes = 5
    d1 = 6
    hid = [600, 300, 100]
elif args.data == 'covertype':
    num_classes = 7
    d1 = 11
    hid = [1000, 600, 300, 100]

def run_exp(d1, num_exp, mask):
    list_train_acc = []
    list_test_acc = []
    list_attack_acc = []
    for iter_exp in range(num_exp):
        net = Net(d1, train_X.shape[1] - d1 - 1, num_classes, hid, mask.defense)
        train(net, (train_dataset, train_loader, validation_dataset, validation_loader), verbose=args.verbose)
        train_acc, test_acc, attack_acc, idx = eval(net, (validation_dataset, validation_loader, test_dataset, test_loader), binary_features)
        list_train_acc.append(train_acc)
        list_test_acc.append(test_acc)
        list_attack_acc.append(attack_acc)
    mask.print_info(list_train_acc, list_test_acc, list_attack_acc)

gauss = Gaussian(0.0)
run_exp(d1, args.repeat, gauss)
        
        
