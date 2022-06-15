#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version: 220615a

# std libs
import argparse, sys, os, time, numpy as np, random
from tqdm import tqdm

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Graphcore (GC) additions
import poptorch

# parsed settings
def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')

    # model parsers
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--concM', type=int, default=100,
                        help='conc MNIST to this factor (default: 1)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum in SGD optimizer (default: 0.5)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')

    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training')

    # parallel parsers
    parser.add_argument('--nworker', type=int, default=0,
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    # GC parsers
    """
    Device iteration defines the number of iterations the device should
    run over the data before returning to the user.
    This is equivalent to running the IPU in a loop over that the specified
    number of iterations, with a new batch of data each time. However, increasing
    deviceIterations is more efficient because the loop runs on the IPU directly.
    """
    parser.add_argument('--device-iterations', type=int, default=50,
                        help='check code! (default: 50)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs

# network
class Block(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              num_filters,
                              kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = Block(1, 10, 5, 2)
        self.layer2 = Block(10, 20, 5, 2)
        self.layer3 = nn.Linear(320, 50)
        self.layer3_act = nn.ReLU()
        self.layer3_dropout = torch.nn.Dropout(0.5)
        self.layer4 = nn.Linear(50, 10)
        # GC - loss is defined in the network
        self.loss = nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 320)
        x = self.layer3_act(self.layer3(x))
        x = self.layer4(self.layer3_dropout(x))
        x = nn.functional.log_softmax(x)
        if self.training:
            return x, self.loss(x, labels)
        return x

# train loop - GC
def train(model, train_loader, epoch):
    model.train()
    t_list = []
    loss_acc=0
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.perf_counter()
        pred,loss = model(data,target)
        if batch_idx % args.log_int == 0:
            print(
                f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\tLoss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
        loss_acc+= loss.item()
    print('TIMER: train time', sum(t_list) / len(t_list),'s')
    return loss_acc

# test loop - GC
def test(model, test_loader):
    model.eval()
    test_loss = 0
    for data, labels in test_loader:
        output = model(data)
        test_loss += accuracy(output, labels)
    print('Accuracy on test set: {:0.2f}%'.format(test_loss / len(test_loader)),'\n')

def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    labels = labels[-predictions.size()[0]:]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy

# save state of the training
def save_state(model,res_name,is_best):
    if is_best:
        rt = time.time()
        torch.save(model.state_dict(),'./'+res_name)
        print(f'DEBUG: state is saved')

# main
def main():
    # get parse args
    pars_ini()

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

    # deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    # some debug
    print('TIMER: initialise:', time.time()-st, 's')
    print('DEBUG: sys.version:',sys.version,'\n')

    print('DEBUG: IO parsers:')
    print('DEBUG: args.data_dir:',args.data_dir)
    print('DEBUG: args.restart_int:',args.restart_int,'\n')

    print('DEBUG: model parsers:')
    print('DEBUG: args.batch_size:',args.batch_size)
    print('DEBUG: args.test_batch_size:',args.test_batch_size)
    print('DEBUG: args.epochs:',args.epochs)
    print('DEBUG: args.lr:',args.lr)
    print('DEBUG: args.concM:',args.concM)
    print('DEBUG: args.momentum:',args.momentum)
    print('DEBUG: args.shuff:',args.shuff,'\n')

    print('DEBUG: debug parsers:')
    print('DEBUG: args.testrun:',args.testrun)
    print('DEBUG: args.nseed:',args.nseed)
    print('DEBUG: args.log_int:',args.log_int,'\n')

    print('DEBUG: parallel parsers:')
    print('DEBUG: args.nworker:',args.nworker)
    print('DEBUG: args.prefetch:',args.prefetch)
    print('DEBUG: args.benchrun:',args.benchrun,'\n')

    print('DEBUG: GC parsers:')
    print('DEBUG: args.device_iterations:',args.device_iterations,'\n')

# load datasets
    data_dir = args.data_dir
    mnist_scale = args.concM
    largeData = []
    for i in range(mnist_scale):
        largeData.append(
            datasets.MNIST(data_dir, train=True, download=False,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
            )

    # concat data
    training_dataset = torch.utils.data.ConcatDataset(largeData)

    mnist_scale = args.concM
    largeData = []
    for i in range(mnist_scale):
        largeData.append(
            datasets.MNIST(data_dir, train=False, download=False,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
            )

    # concat data
    test_dataset = torch.utils.data.ConcatDataset(largeData)

# GC - set training options
    """
    To accelerate the training deviceIterations=50 is set
    data loader will pick 50 batches of data per step.
    """
    training_opts = poptorch.Options()
    training_opts.deviceIterations(args.device_iterations)

# GC - data loader provided by PopTorch
    args.shuff = args.shuff and not args.testrun
    train_loader = poptorch.DataLoader(
        options=training_opts,
        dataset=training_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuff,
        drop_last=True,
        num_workers=args.nworker
    )

    """
    A `poptorch.Options()` instance contains a set of default hyperparameters and options for the IPU.
    """
    test_loader = poptorch.DataLoader(
        options=poptorch.Options(),
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.nworker
    )

    print('TIMER: read and concat data:', time.time()-st, 's') 

# create CNN model
    model = Network()

# optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# GC - distribute model to IPU
    train_model = poptorch.trainingModel(
        model,
        training_opts,
        optimizer=optimizer
    )

# GC - distribute model to IPU w/o training options (for testing)
    test_model = poptorch.inferenceModel(model,options=poptorch.Options())

# resume state if any
    best_acc = np.Inf
    res_name='checkpoint.pth.tar'
    start_epoch = 1
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            checkpoint = torch.load(program_dir+'/'+res_name)
            start_epoch = checkpoint['epoch']
            print(f'WARNING: restarting from {start_epoch} epoch')
        except:
            print(f'WARNING: restart file cannot be loaded, restarting!')

    if start_epoch>=args.epochs+1:
        print(f'WARNING: given epochs are less than the one in the restart file!\n' 
              f'WARNING: SYS.EXIT is issued')
        sys.exit()

# start trainin/testing loop
    print('TIMER: initialization:', time.time()-st, 's')
    print(f'\nDEBUG: start training')
    print(f'--------------------------------------------------------') 

    et = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lt = time.time()

        # GC - combines forward + backward 
        loss_acc = train(train_model, train_loader, epoch)

        # GC - testing
        acc_test = test(test_model, test_loader)

        # save first epoch timer
        if epoch == start_epoch:
            first_ep_t = time.time()-lt

        print('TIMER: epoch time:', time.time()-lt, 's')

# GC - unload models from IPU
    train_model.detachFromDevice()
    test_model.detachFromDevice()
    
# save final state
    if not args.benchrun:
        save_state(train_model,res_name,True)

    # some debug
    print(f'\n--------------------------------------------------------')
    print('DEBUG: training results:\n')
    print('TIMER: first epoch time:', first_ep_t, ' s')
    print('TIMER: last epoch time:', time.time()-lt, ' s')
    print('TIMER: average epoch time:', (time.time()-et)/args.epochs, ' s')
    print('TIMER: total epoch time:', time.time()-et, ' s')
    if epoch > 1:
        print('TIMER: total epoch-1 time:', time.time()-et-first_ep_t, ' s')
        print('TIMER: average epoch-1 time:', (time.time()-et-first_ep_t)/(args.epochs-1), ' s')
    if args.benchrun:
        print('TIMER: total epoch-2 time:', lt-first_ep_t, ' s')
        print('TIMER: average epoch-2 time:', (lt-first_ep_t)/(args.epochs-2), ' s')

if __name__ == "__main__": 
    main()
    sys.exit()

#eof
