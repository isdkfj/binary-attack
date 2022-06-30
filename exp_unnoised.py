from utils import get_args, set_random_seed
from data import load_data
from train import prepare_dataset, train
from eval_naive import eval
from module import Net

args = get_args()
set_random_seed(args.seed)

# last column of X is fabricated label
train_X, test_X, train_Y, test_Y = load_data(args.data, args.path, args.seed)
train_dataset, train_loader, test_dataset, test_loader = prepare_dataset(train_X, train_Y, test_X, test_Y, args.bs)

num_classes = 2
if args.data == 'nursery':
    num_classes = 5
elif args.data == 'covertype':
    num_classes = 7

def run_exp(d1, num_exp, mask):
    sum_train_acc = 0
    sum_test_acc = 0
    sum_attack_acc = 0
    for iter_exp in range(num_exp):
        net = Net(d1, train_X.shape[1] - d1 - 1, num_classes, args.net, mask.defense)
        #train(net, (train_dataset, train_loader, test_dataset, test_loader), verbose=True)
        eval(net, (train_dataset, train_loader, test_dataset, test_loader))
dimensions = [8]
if args.data == 'mushroom':
    dimensions = [15]
elif args.data == 'nursery':
    dimensions = [6]
elif args.data == 'covertype':
    dimensions = [11]
for d1 in dimensions:
    gauss = Gaussian(0.0)
    run_exp(d1, 1, gauss)
        
        
