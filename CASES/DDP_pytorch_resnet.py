import argparse, sys, platform, os, time, numpy as np
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

# training settings
def parsIni():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--backend', type=str, default='nccl', metavar='N',
                        help='backend for parrallelisation (default: nccl)')
    return parser

Net = models.resnet50(False)
Net.avgpool = nn.AdaptiveAvgPool2d(1)
Net.fc.out_features = 200

def test(model, device, test_loader, lrank, grank, nworker):
    model.eval()
    test_loss = 0
    correct = 0
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if grank==0 and lrank==0:
        print(
            f'Test set: average loss: {test_loss:.4f}\t'
            f'accurate samples: {correct}/{len(test_loader.dataset)/nworker}')
    acc_test = 100.0 * correct * nworker / len(test_loader.dataset)
    return acc_test

def train(args, model, device, train_loader, optimizer, epoch, lrank, grank, nworker):
    model.train()
    t_list = []
    if grank==0 and lrank==0:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.perf_counter()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and grank==0 and lrank==0:
            print(
                f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)/nworker} '
                f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\tLoss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
    if grank==0 and lrank==0:
        print('TIMER: train time', sum(t_list) / len(t_list),'s')
#
#
# MAIN
#
#
def main():
    # get parse args
    parser = parsIni()
    args = parser.parse_args()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    # initializes the distributed backend which will take care of sychronizing nodes/GPUs
    st = timer()
    torch.distributed.init_process_group(backend=args.backend, init_method='env://')
    grank = int(os.environ.get("SLURM_PROCID", 0))
    lrank = torch.distributed.get_rank()
    nnode = int(os.environ.get("SLURM_NNODES", 0))
    wsize = torch.distributed.get_world_size()
    nworker = wsize*nnode
    print(f'local rank:', lrank, 'global rank', grank)
    torch.distributed.barrier()

    # some debug
    if grank==0 and lrank==0: 
        print('TIMER: initialise:', timer()-st, 's') 
        print('DEBUG: sys.version:',sys.version)
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.backend:',args.backend,'\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)

    # Have all workers wait - prints errors at the moment
    torch.distributed.barrier()

    train_data_dir = '/p/project/cslfse/aach1/tiny-imagenet-200/train'
    test_data_dir = '/p/project/cslfse/aach1/tiny-imagenet-200/val'

    train_dataset = datasets.ImageFolder(train_data_dir, transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder(test_data_dir, transform=transforms.ToTensor())

    # restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=nworker, rank=grank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=nworker, rank=grank)

    # load dataset with DDP
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, shuffle=False)

    if grank==0 and lrank==0: 
        print('TIMER: read and concat data:', timer()-st, 's') 

    # create CNN model
    model = Net.to(device)

    # Optimizer step
    #optimizer = torch.optim.Adam(mel.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # distribute
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model,\
            device_ids=[device], output_device=device)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model)

    if grank==0 and lrank==0: 
        print('TIMER: broadcast:', timer()-st, 's') 
        print(f'\nDEBUG: start training') 
        print(f'--------------------------------------------------------') 

    et = timer()
    for epoch in range(1, args.epochs + 1):
        lt = timer()
        train(args, distrib_model, device, train_loader, optimizer, epoch, lrank, grank, nworker)
        acc_test = test(distrib_model, device, test_loader, lrank, grank, nworker)
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True
            test_loader.last_epoch = True
        if grank==0 and lrank==0: 
            print('TIMER: epoch time:', timer()-lt, 's')
            print('DEBUG: accuracy:', acc_test, '%')
        torch.distributed.barrier(async_op=True)

    # finalise
    torch.distributed.barrier()
    if grank==0 and lrank==0: 
        print(f'\n--------------------------------------------------------') 
        print('DEBUG: results:\n') 
        print('TIMER: last epoch time:', timer()-lt, 's') 
        print('TIMER: total epoch time:', timer()-et, 's')
        print('DEBUG: last accuracy:', acc_test, '%')

    if grank==0 and lrank==0 and args.cuda:
        print('DEBUG: memory req:',int(torch.cuda.memory_reserved(lrank)/1024/1024),'MB')
    elif not args.cuda:
        print('DEBUG: memory req: 0 MB')

    print(f'TIMER: final time 1: {timer()-st} s\n')
    time.sleep(60)
    print(f'TIMER: final time 2: {timer()-st} s\n')

if __name__ == "__main__": 
    main()

#eof
