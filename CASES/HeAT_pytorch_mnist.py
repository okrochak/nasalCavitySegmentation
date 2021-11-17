#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: EI
# version: 210701a

# libs
import argparse, sys, platform, os, time, numpy as np
from timeit import default_timer as timer
from filelock import FileLock

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import datasets, transforms

import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.optim.lr_scheduler import StepLR
from heat.utils import vision_transforms
from heat.utils.data.mnist import MNISTDataset

# training settings
def parsIni():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--concM', type=int, default=100, metavar='N',
                        help='conc MNIST to this factor (default: 100)')
    parser.add_argument("--log-interval", type=int, default=100, metavar="N", help="log inter")

    return parser


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net2(ht.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ht.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = ht.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = ht.nn.Dropout2d(0.25)
        self.dropout2 = ht.nn.Dropout2d(0.5)
        self.fc1 = ht.nn.Linear(9216, 128)
        self.fc2 = ht.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    t_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.perf_counter()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
        t_list.append(time.perf_counter() - t)
    print("average time", sum(t_list) / len(t_list))

# gathers any object from the whole group in a list (to all workers)
def par_allgather_obj(obj,gwsize):
    res = [None]*gwsize
    dist.all_gather_object(res,obj,group=None)
    return res
#
#
# MAIN
#
#
if __name__ == '__main__':
    # get parse args
    parser = parsIni()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # init
    st = timer()
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # some debug
    if ht.MPI_WORLD.rank==0:
        print('DEBUG: sys.version:',sys.version)
        print('DEBUG: torch.cuda.is_available():',torch.cuda.is_available())
        print('DEBUG: torch.cuda.current_device():',torch.cuda.current_device())
        print('DEBUG: torch.cuda.device_count():',torch.cuda.device_count())
        print('DEBUG: torch.cuda.get_device_properties(0):',
            torch.cuda.get_device_properties(0))
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.concM:',args.concM,'\n')

    if ht.MPI_WORLD.rank==0:
        print('TIMER: initialise:', timer()-st, 's') 

    if args.cuda:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(args.seed)

    kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        kwargs.update({"num_workers": 0, "pin_memory": True})

    # read and/or generate dataset
    transform = ht.utils.vision_transforms.Compose(
        [vision_transforms.ToTensor(), vision_transforms.Normalize((0.1307,), (0.3081,))]
    )

    # train dataset
    data_dir = args.data_dir
    
    # # THIS PART NEEDS ATTENTION
    # mnist_scale = args.concM
    # largeData1 = []
    # largeData2 = []
    # for i in range(mnist_scale):
    #     largeData1.append(MNISTDataset(data_dir, train=True, transform=transform, 
    #         download=False, ishuffle=False))
    #     largeData2.append(MNISTDataset(data_dir, train=True, transform=transform, 
    #         download=False, ishuffle=False, test_set=True))
    # # concat data
    # #train_dataset = ht.utils.data.datatools.torch_data.ConcatDataset(largeData1)
    # #test_dataset  = ht.utils.data.datatools.torch_data.ConcatDataset(largeData2)
    # print(largeData1)
    # train_dataset = np.concatenate([largeData1],axis=0)
    # test_dataset  = np.concatenate([largeData2],axis=0)
    
    # append data does not work... going for STD
    train_dataset = MNISTDataset(data_dir, train=True, transform=transform, 
        download=False, ishuffle=False)
    test_dataset = MNISTDataset(data_dir, train=False, transform=transform, 
        download=False, ishuffle=False, test_set=True)
    
    # distributre dataset with heat
    train_loader = ht.utils.data.datatools.DataLoader(dataset=train_dataset, **kwargs)
    test_loader = ht.utils.data.datatools.DataLoader(dataset=test_dataset, **kwargs)

    if ht.MPI_WORLD.rank==0:
        print('TIMER: read and concat data:', timer()-st, 's') 

    # create CNN model
    model = Net().to(device)

    # Optimizer step 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # distribute
    blocking = False
    dp_optim = ht.optim.DataParallelOptimizer(optimizer, blocking=blocking)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    dp_model = ht.nn.DataParallel(
        model, comm=train_dataset.comm, optimizer=dp_optim, blocking_parameter_updates=blocking
    )

    if ht.MPI_WORLD.rank==0:
        print('TIMER: broadcast:', timer()-st, 's') 
    
    et = timer()
    for epoch in range(1, args.epochs + 1):
        lt = timer()
        train(args, dp_model, device, train_loader, dp_optim, epoch)
        test(dp_model, device, test_loader)
        scheduler.step()
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True
            test_loader.last_epoch = True
        if ht.MPI_WORLD.rank==0:
            print('TIMER: epoch time:', timer()-lt, 's') 
    print('TIMER: last epoch time:', timer()-lt, 's') 
    print('TIMER: total epoch time:', timer()-et, 's') 

    if ht.MPI_WORLD.rank==0:
        print('\n',torch.cuda.memory_summary(0),'\n')

    if ht.MPI_WORLD.rank==0:
        print('DEBUG: memory req:',
            int(torch.cuda.memory_reserved(0)/1024/1024),'MB')

    # not working atm
    #is_best=ht.MPI_WORLD.rank
    #gwsize=4
    #isb = par_allgather_obj(is_best,gwsize)
    #print(f'test: {isb}\n')

# eof
