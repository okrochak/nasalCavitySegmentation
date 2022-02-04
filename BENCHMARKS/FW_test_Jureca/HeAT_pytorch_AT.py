#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version: 211005a

# std libs
import argparse, sys, platform, os, time, numpy as np, h5py, random, shutil
import matplotlib.pyplot as plt

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate, pad
from torchvision import datasets, transforms

import heat as ht
import heat.nn.functional as F
import heat.optim as optim
from heat.optim.lr_scheduler import StepLR
from heat.utils import vision_transforms

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    data_u = torch.from_numpy(np.array(f[list(f.keys())[0]]['u'])).permute((1,0,2))
    data_v = torch.from_numpy(np.array(f[list(f.keys())[0]]['v'])).permute((1,0,2))
    data_w = torch.from_numpy(np.array(f[list(f.keys())[0]]['w'])).permute((1,0,2))
    return torch.reshape(torch.cat((data_u, data_v, data_w)),
                    (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3))

# training settings
def pars_ini():
    parser = argparse.ArgumentParser(description='PyTorch AutoEncoder Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-int', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--restart-int', type=int, default=10, metavar='N',
                        help='restart int per epoch (default: 10)')
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run (default: False)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a test run (default: False)')
    parser.add_argument('--nworker', type=int, default=0,
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader - not implemented (default: 2)')
    return parser

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.reLU = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.tanhf = nn.Tanh()

#Encoding layers - conv_en1 denotes 1st (1) convolutional layer for the encoding (en) part
        self.conv_en1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bnorm_en1 = nn.BatchNorm2d(16)
        self.conv_en2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bnorm_en2 = nn.BatchNorm2d(32)
        #self.conv_en3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=[2,1])
        #self.bnorm_en3 = nn.BatchNorm2d(64)

#Decoding layers - conv_de1 denotes outermost (1) convolutional layer for the decoding (de) part
        #self.bnorm_de3 = nn.BatchNorm2d(64)
        #self.conv_de3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=[0,1])
        self.bnorm_de2 = nn.BatchNorm2d(32)
        # for 3 layer compression, use the following for the 2nd deconvolution
        #self.conv_de2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=[0,1])
        # for 2 layer compression, use the following for the 2nd deconvolution
        self.conv_de2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=[1,1])
        self.bnorm_de1 = nn.BatchNorm2d(16)
        # for 3 layer compression, use the following for the 1st deconvolution
        #self.conv_de1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=[3,1])
        # for 2 layer compression, use the following for the 1st deconvolution
        self.conv_de1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=[0,1])

    def forward(self, x):
        """
        encoder - Convolutional layer is followed by a reLU activation which is then batch normalised
        """
# 1st encoding layer
        x = self.conv_en1(x)
        x = self.reLU(x)
        x = self.bnorm_en1(x)
# 2nd encoding layer
        x = self.conv_en2(x)
        x = self.reLU(x)
        x = self.bnorm_en2(x)
# 3rd encoding layer
        #x = self.conv_en3(x)
        #x = self.reLU(x)
        #x = self.bnorm_en3(x)
        #x = torch.flatten(x, start_dim=1)

        """
        Decoder:
        Map the given latent code to the image space.
        decoder - reLU activation is applied first followed by a batch normalisation 
        and a convolutional layer which maintains the same dimension and an interpolation 
        layer in the end to decompress the dataset
        """
# 3rd decoding layer
        #x = self.reLU(x)
        #x = self.bnorm_de3(x)
        #x = self.conv_de3(x)
        #x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
# 2nd decoding layer
        x = self.reLU(x)
        x = self.bnorm_de2(x)
        x = self.conv_de2(x)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
# 1st decoding layer
        x = self.reLU(x)
        x = self.bnorm_de1(x)
        x = self.conv_de1(x)
        x = pad(x,(0,0,0,1))
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.tanhf(x)

# PARALLEL HELPERS
# sum of field over GPGPUs
def par_sum(field,hmp):
    res = torch.tensor(field).float().cuda()
    hmp.Allreduce(res,res)
    return res

# mean of field over GPGPUs
def par_mean(field,hmp,gwsize):
    res = torch.tensor(field).float().cuda()
    hmp.Allreduce(res,res)
    res/=gwsize
    return res
#
#
# MAIN
#
#
def main():
    # get parse args
    parser = pars_ini()
    args = parser.parse_args()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # limit # of CPU threads to be used per worker
    torch.manual_seed(args.seed)

    # get directory
    program_dir = os.getcwd()
    
    # start the time.time for profiling
    st = time.time()
  
    # get job rank info - rank==0 master gpu
    gwsize = ht.MPI_WORLD.size         # global world size - per run
    lwsize = torch.cuda.device_count() # local world size - per node
    grank = ht.MPI_WORLD.rank          # global rank - assign per run
    lrank = grank%lwsize               # local rank - assign per node

    # Init HeAT MPI comm
    htMPI = ht.MPICommunication()

    # some debug
    if ht.MPI_WORLD.rank==0:
        print('TIMER: initialise:', time.time()-st, 's') 
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:',sys.version)
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.lr:',args.lr)
        print('DEBUG: args.log_int:',args.log_int)
        print('DEBUG: args.restart_int:',args.restart_int)
        print('DEBUG: args.cuda:',args.cuda)
        print('DEBUG: args.nworker:',args.nworker)
        print('DEBUG: args.prefetch:',args.prefetch)
        print('DEBUG: args.testrun:',args.testrun)
        print('DEBUG: args.benchrun:',args.benchrun,'\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)

# load datasets
    transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    #loc1 = args.data_dir+'trainfolder'
    turb_data = datasets.DatasetFolder(args.data_dir+'trainfolder',\
        loader=hdf5_loader, transform=transform, extensions='.hdf5')
    test_data = datasets.DatasetFolder(args.data_dir+'testfolder',\
         loader=hdf5_loader, transform=transform, extensions='.hdf5')

    # restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        turb_data, num_replicas=gwsize, rank=lrank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data, num_replicas=gwsize, rank=lrank)

# distribute dataset to workers
    train_loader = torch.utils.data.DataLoader(turb_data, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.nworker, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.nworker, pin_memory=True, shuffle=False)
    #train_loader = ht.utils.data.datatools.DataLoader(dataset=turb_data, **kwargs)

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n') 

    # create model
    model = autoencoder().to(device)

# optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.003)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# distribute
    blocking = False
    dp_optimizer = ht.optim.DataParallelOptimizer(optimizer, blocking=blocking)
    distrib_model = ht.nn.DataParallel(
        model, comm=ht.get_comm(), optimizer=dp_optimizer, blocking_parameter_updates=blocking)
    #distrib_model = ht.nn.DataParallelMultiGPU(
    #    model, optimizer=dp_optimizer)

# start trainin loop
    et = time.time()
    for epoch in range(args.epochs):
        lt = time.time()
        loss_acc = 0.0
        count=0
        train_sampler.set_epoch(epoch)
        for data in train_loader:
            inputs = data[:-1][0]
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0).to(device)
            # ===================forward=====================
            dp_optimizer.zero_grad()
            predictions = distrib_model(inputs)         # Forward pass
            loss = loss_function(predictions.float(), inputs.float())   # Compute loss function
            # ===================backward====================
            loss.backward()                                        # Backward pass
            dp_optimizer.step()                                    # Optimizer step
            loss_acc+= loss.item()
            if count % args.log_int == 0 and grank==0:
                print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):.2f}% complete',\
                      f' / {time.time() - lt:.2f} s / accumulated loss: {loss_acc}\n')
            count+=1

        if grank==0:
            print('TIMER: epoch time:', time.time()-lt, 's')

        #scheduler_lr.step()

    # some debug
    if grank==0:
        print(f'\n--------------------------------------------------------') 
        print('DEBUG: training results:\n')
        print(f'TIMER: last epoch time: {time.time()-lt} s\n')
        print(f'TIMER: total epoch time: {time.time()-et} s\n')
        if args.cuda:
            print('DEBUG: memory req:',int(torch.cuda.memory_reserved(lrank)/1024/1024),'MB')
        elif not args.cuda:
            print('DEBUG: memory req: - MB')

# start testing loop
    et = time.time()
    distrib_model.eval()
    test_loss = 0.0
    mean_sqr_diff = []
    count=0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[:-1][0]
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0).to(device)
            predictions = distrib_model(inputs)
            loss = loss_function(predictions.float(), inputs.float())
            test_loss+= loss.item()/inputs.shape[0]
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            mean_sqr_diff.append(\
                torch.mean(torch.square(predictions.float()-inputs.float())).item())
            count+=1

    # mean from dataset (ignore if just 1 dataset)
    if count>1:
        mean_sqr_diff=np.mean(mean_sqr_diff)
    if grank==0:
        print(f'TIMER: total testing time: {time.time()-et} s\n')

# finalise testing
    # mean from gpus
    avg_mean_sqr_diff = par_mean(mean_sqr_diff,htMPI,gwsize)
    avg_mean_sqr_diff = float(np.float_(avg_mean_sqr_diff.data.cpu().numpy()))
    if grank==0:
        print(f'DEBUG: avg_mean_sqr_diff: {avg_mean_sqr_diff}\n')

# clean-up
    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s\n')

if __name__ == "__main__": 
    main()
    sys.exit()

# eof
