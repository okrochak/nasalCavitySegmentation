#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version: 210916a

# std libs
import argparse, sys, os, time, numpy as np

# ml libs
import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# training settings
def parsIni():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log-int', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--concM', type=int, default=100, metavar='N',
                        help='conc MNIST to this factor (default: 100)')
    parser.add_argument('--backend', type=str, default='nccl', metavar='N',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--restart-int', type=int, default=10, metavar='N',
                        help='restart int per epoch (default: 10)')
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run (default: False)')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    # parse to deepspeed
    parser = deepspeed.add_config_arguments(parser)
    return parser

# network
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

# train loop
def train(args, model, train_loader, optimizer, epoch, grank, gwsize):
    device = model.local_rank
    t_list = []
    loss_acc=0
    if grank==0:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.perf_counter()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_int == 0 and grank==0:
            print(
                f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)/gwsize} '
                f'({100.0 * batch_idx *len(data) / len(train_loader):.0f}%)]\t\tLoss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
        loss_acc+= loss.item()
    if grank==0:
        print('TIMER: train time', sum(t_list) / len(t_list),'s')
    return loss_acc

# test loop
def test(model, test_loader, grank, gwsize):
    device = model.local_rank
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
    if grank==0:
        print(
            f'Test set: average loss: {test_loss:.4f}\t'
            f'accurate samples: {correct}/{len(test_loader.dataset)/gwsize}')
    acc_test = 100.0 * correct * gwsize / len(test_loader.dataset)
    return acc_test

# save state of the training
def save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,is_best):
    rt = time.time()
    # find if is_best happened in any worker
    is_best_m = par_allgather_obj(is_best,gwsize)

    if any(is_best_m):
        # find which rank is_best happened - select first rank if multiple
        is_best_rank = np.where(np.array(is_best_m)==True)[0][0]

        # collect state
        state = {'epoch': epoch + 1,
                 'state_dict': distrib_model.state_dict(),
                 'best_acc': loss_acc,
                 'optimizer' : optimizer.state_dict()}

        # write on worker with is_best
        if grank == is_best_rank: 
            torch.save(state,'./'+res_name)
            print(f'DEBUG: state in {grank} is saved on epoch:{epoch} in {time.time()-rt} s')

# PARALLEL HELPERS
# sum of field over GPGPUs
def par_sum(field):
    res = torch.tensor(field).float().cuda()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    return res

# mean of field over GPGPUs
def par_mean(field,gwsize):
    res = torch.tensor(field).float().cuda()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    res/=gwsize
    return res

# max(field) over GPGPUs
def par_max(field):
    res = torch.tensor(field).float().cuda()
    dist.all_reduce(res,op=dist.ReduceOp.MAX,group=None,async_op=True).wait()
    return res

# min(field) over GPGPUs
def par_min(field):
    res = torch.tensor(field).float().cuda()
    dist.all_reduce(res,op=dist.ReduceOp.MIN,group=None,async_op=True).wait()
    return res

# reduce field to destination with an operation
def par_reduce(field,dest,oper):
    '''
    dest=0 will send the result to GPU on rank 0 (any rank is possible)
    op=oper has to be in form "dist.ReduceOp.<oper>", where <oper> is
      SUM
      PRODUCT
      MIN
      MAX
      BAND
      BOR
      BXOR
    '''
    res = torch.Tensor([field]).cuda()
    dist.reduce(res,dst=dest,op=oper,group=None,async_op=False)
    return res.item()

# gathers tensors from the whole group in a list (to all workers)
def par_allgather(field,gwsize):
    sen = torch.Tensor([field]).cuda()
    res = [torch.Tensor([field]).cuda() for i in range(gwsize)]
    dist.all_gather(res,sen,group=None)
    return res

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
def main():
    # get parse args
    parser = parsIni()
    args = parser.parse_args()

    # limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

# initializes the distributed backend which will take care of sychronizing nodes/GPUs
    deepspeed.init_distributed(dist_backend=args.backend)

    # get job rank info - rank==0 master gpu
    gwsize = dist.get_world_size()     # global world size - per run
    lwsize = torch.cuda.device_count() # local world size - per node
    grank = dist.get_rank()            # global rank - assign per run
    lrank = dist.get_rank()%lwsize     # local rank - assign per node

    # some debug
    if grank==0: 
        print('TIMER: initialise:', time.time()-st, 's') 
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:',sys.version)
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.lr:',args.lr)
        print('DEBUG: args.concM:',args.concM)
        print('DEBUG: args.backend:',args.backend)
        print('DEBUG: args.log_int:',args.log_int)
        print('DEBUG: args.restart_int:',args.restart_int)
        print('DEBUG: args.benchrun:',args.benchrun)
        print('DEBUG: args.testrun:',args.testrun,'\n')

    # encapsulate the model on the GPU assigned to the current process
    torch.cuda.set_device(lrank)

# read training dataset
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
    train_dataset = torch.utils.data.ConcatDataset(largeData)

# read test dataset
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

# distribute test dataset
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=gwsize, rank=grank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=0, pin_memory=True, shuffle=False)

    if grank==0:
        print('TIMER: read and concat data:', time.time()-st, 's') 

# create CNN model
    model = Net()

# Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) DeepSpeed optimizer
    # 3) Distributed data loader
    distrib_model, __, train_loader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(), training_data=train_dataset)

# optimizer
    #optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr, momentum=0.5)

# resume state 
    start_epoch = 1
    best_acc = np.Inf 
    res_name='checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank}
            checkpoint = torch.load(program_dir+'/'+res_name, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            distrib_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if grank==0:
                print(f'WARNING: restarting from {start_epoch} epoch')
        except:
            if grank==0:
                print(f'WARNING: restart file cannot be loaded, restarting!')

    if start_epoch>=args.epochs+1:
        if grank==0:
            print(f'WARNING: given epochs are less than the one in the restart file!\n' 
                  f'WARNING: SYS.EXIT is issued')
        deepspeed.sys.exit()
        sys.exit()

# start trainin/testing loop
    if grank==0:
        print('TIMER: broadcast:', time.time()-st, 's') 
        print(f'\nDEBUG: start training') 
        print(f'--------------------------------------------------------') 

    et = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lt = time.time()
        # training
        loss_acc = train(args, distrib_model, train_loader, optimizer, epoch, grank, gwsize)

        # testing
        acc_test = test(distrib_model, test_loader, grank, gwsize)

        # save state if found a better state 
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0 and not args.benchrun:
            save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,is_best)
            # reset best_acc
            best_acc = min(loss_acc, best_acc)

        if grank==0:
            print('TIMER: epoch time:', time.time()-lt, 's')
            print('DEBUG: accuracy:', acc_test, '%')

# finalise
    # save final state
    if not args.benchrun:
        save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,True)
    dist.barrier()

    # some debug
    if grank==0:
        print(f'\n--------------------------------------------------------') 
        print('DEBUG: results:\n')
        print('TIMER: last epoch time:', time.time()-lt, 's') 
        print('TIMER: total epoch time:', time.time()-et, 's')
        print('DEBUG: last accuracy:', acc_test, '%')
        print('DEBUG: memory req:',int(torch.cuda.memory_reserved(lrank)/1024/1024),'MB')

    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s\n')

# clean-up
    deepspeed.sys.exit()

if __name__ == "__main__": 
    main()
    sys.exit()

#eof
