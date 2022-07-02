import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import accuracy

def prepare_dataset(train_X, train_Y, test_X, test_Y, batch_size):
    class TensorDataset(Dataset):
        def __init__(self, data_tensor, target_tensor):
            self.data_tensor = data_tensor
            self.target_tensor = target_tensor

        def __getitem__(self, index):
            return self.data_tensor[index], self.target_tensor[index]

        def __len__(self):
            return self.data_tensor.size(0)

    train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_Y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_Y))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_dataset, train_loader, test_dataset, test_loader

def train(net, data, verbose=False):
    train_dataset, train_loader= data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    num_epoch = 100

    for epoch in range(1, num_epoch + 1):
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            if epoch <= 10:
                print(loss.item())
            nn.utils.clip_grad_value_(net.parameters(), 2)
            optimizer.step()
        scheduler.step()
        if epoch % 1 == 0 and epoch <= 10 and verbose:
            with torch.no_grad():
                total_loss = 0.0
                total_acc = 0.0
                for i, (data, target) in enumerate(train_loader):
                    output = net(data)
                    loss = criterion(output, target)
                    total_loss += loss.item() * len(data)
                    total_acc += accuracy(output, target).item() * len(data)
                total_loss /= len(train_dataset)
                total_acc /= len(train_dataset)
                print('epoch {} train loss:'.format(epoch), total_loss)
                print('epoch {} train acc:'.format(epoch), total_acc)
