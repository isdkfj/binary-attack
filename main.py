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
train_dataset, train_loader, test_dataset, test_loader = prepare_dataset(train_X, train_Y, test_X, test_Y, args.bs)

def run_exp(D1, num_exp, mask):
    sum_train_acc = 0
    sum_test_acc = 0
    sum_attack_acc = 0
    for iter_exp in range(num_exp):
        net = Net(D1, train_X.shape[1] - D1 - 1, 2, args.net, mask.defense)
        print(net)
        train(net, (train_dataset, train_loader, test_dataset, test_loader), verbose=True)
        train_acc, test_acc, attack_acc, idx = eval(net, (train_dataset, train_loader, test_dataset, test_loader))
        sum_train_acc += train_acc
        sum_test_acc += test_acc
        sum_attack_acc += attack_acc
        print(train_acc, test_acc, attack_acc)
    mask.print_info(sum_train_acc / num_exp, sum_test_acc / num_exp, sum_attack_acc / num_exp)

for D1 in [8]:
    '''gauss = Gaussian(0.0)
    run_exp(D1, 10, gauss)
    gauss = Gaussian(0.01)
    run_exp(D1, 10, gauss)
    gauss = Gaussian(0.05)
    run_exp(D1, 10, gauss)
    gauss = Gaussian(0.1)
    run_exp(D1, 10, gauss)
    gauss = Gaussian(0.5)
    run_exp(D1, 10, gauss)'''
    fab = Defense(D1)
    run_exp(D1, 10, fab)
        
        
