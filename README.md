# Feature Reconstruction Attacks and Countermeasures of DNN training in Vertical Federated Learning

Usage: ```python main.py``` + options

Options:
- path: the directory to the dataset
- data: specify which dataset to use
- net: number of neurons in each hidden layer
- bs: batch size
- seed: set random seed
- repeat: number of trials
- verbose: whether to print detailed information (e.g., training accuracy and loss in each iteration)
- am: specify which attack method to use
- dm: specify which defense method to use
- nf: set number of fabiricated features
- eps: level of noise added in the forward pass

For example, put the dataset files in the fold ```./data```, then run ```python main.py --path ./data --data bank --net [200, 100] --am regression``` to test the attack results on bank dataset. The first layer of the neuron network contains 200 neurons and the second layer contains 100 layers.
