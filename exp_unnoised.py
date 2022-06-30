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
    dimensions = [8]
    hid = [60, 30, 10]
elif args.data =='credit':
    num_classes = 2
    dimensions = [10]
    hid = [50, 20, 10]
elif args.data == 'mushroom':
    num_classes = 2
    dimensions = [15]
    hid = [50, 20]
elif args.data == 'nursery':
    num_classes = 5
    dimensions = [6]
    hid = [20, 10]
elif args.data == 'covertype':
    num_classes = 7
    dimensions = [11]
    hid = args.net

def run_exp(d1, num_exp, mask):
    sum_train_acc = 0
    sum_test_acc = 0
    sum_attack_acc = 0
    for iter_exp in range(num_exp):
        net = Net(d1, train_X.shape[1] - d1 - 1, num_classes, hid, mask.defense)
        train(net, (train_dataset, train_loader), verbose=False)
        train_acc, test_acc, attack_acc, idx = eval(net, (train_dataset, train_loader, test_dataset, test_loader))
        sum_train_acc += train_acc
        sum_test_acc += test_acc
        sum_attack_acc += attack_acc
        print(test_acc, attack_acc)
    mask.print_info(sum_train_acc / num_exp, sum_test_acc / num_exp, sum_attack_acc / num_exp)

for d1 in dimensions:
    gauss = Gaussian(0.0)
    run_exp(d1, 10, gauss)
        
        
