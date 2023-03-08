#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: EI
# version: 211005a

# libs
import argparse, sys, platform, os, time, numpy as np, h5py, random, shutil
from timeit import default_timer as timer
from filelock import FileLock

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate, pad
from torchvision import datasets, transforms

import horovod.torch as hvd

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    #data_u = torch.FloatTensor(f[list(f.keys())[0]]['u']).permute((1,0,2))
    #data_v = torch.FloatTensor(f[list(f.keys())[0]]['v']).permute((1,0,2))
    #data_w = torch.FloatTensor(f[list(f.keys())[0]]['w']).permute((1,0,2))
    data_u = torch.from_numpy(np.array(f[list(f.keys())[0]]['u'])).permute((1,0,2))
    data_v = torch.from_numpy(np.array(f[list(f.keys())[0]]['v'])).permute((1,0,2))
    data_w = torch.from_numpy(np.array(f[list(f.keys())[0]]['w'])).permute((1,0,2))
    return torch.reshape(torch.cat((data_u, data_v, data_w)),
                    (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3))

# training settings
def pars_ini():
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
    parser.add_argument('--log-int', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--restart-int', type=int, default=10, metavar='N',
                         help='restart int per epoch (default: 10)')
    parser.add_argument('--data-dir', default='/p/project/prcoe12/RAISE/T31',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a test run (default: False)')
    parser.add_argument('--concM', type=int, default=100, metavar='N',
                        help='conc MNIST to this factor (default: 100)')
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

# sum of field over GPGPUs
def par_mean(field):
    res = torch.tensor(field).float().cuda()
    hvd.allreduce_(res, average=False)
    return res

# mean of field over GPGPUs
def par_mean(field):
    res = torch.tensor(field).float().cuda()
    hvd.allreduce_(res, average=True)
    return res

# gathers any object from the whole group in a list (to all workers)
def par_allgather_obj(obj):
    return hvd.allgather_object(obj)
#
#
# MAIN
#
#
def main():
    # get parse args
    parser = pars_ini()
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: init
    st = timer()
    hvd.init()
    #torch.manual_seed(args.seed)

    # some debug
    if hvd.rank()==0 and hvd.local_rank()==0:
        print('TIMER: initialise:', timer()-st, 's') 
        print('DEBUG: sys.version:',sys.version)
        print('DEBUG: torch.cuda.is_available():',torch.cuda.is_available())
        print('DEBUG: torch.cuda.current_device():',torch.cuda.current_device())
        print('DEBUG: torch.cuda.device_count():',torch.cuda.device_count())
        print('DEBUG: torch.cuda.get_device_properties(hvd.local_rank()):',
            torch.cuda.get_device_properties(hvd.local_rank()))
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.lr:',args.lr)
        print('DEBUG: args.log_int:',args.log_int)
        print('DEBUG: args.restart_int:',args.restart_int)
        print('DEBUG: args.nworker:',args.nworker)
        print('DEBUG: args.prefetch:',args.prefetch)
        print('DEBUG: args.benchrun:',args.benchrun,'\n')

    if args.cuda:
        # Horovod: pin GPU to local rank
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker
    #torch.set_num_threads(1)

    kwargs = {'num_workers': args.nworker, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead... 
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    train_dataset = datasets.DatasetFolder(args.data_dir+'trainfolder', loader=hdf5_loader, \
        transform=transform, extensions='.hdf5')
    test_dataset = datasets.DatasetFolder(args.data_dir+'testfolder', loader=hdf5_loader, \
        transform=transform, extensions='.hdf5')

    # Horovod: use DistributedSampler to partition the training data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, **kwargs)

    if hvd.rank()==0 and hvd.local_rank()==0:
        print('TIMER: read and concat data:', timer()-st, 's') 

    # create CNN model
    model = autoencoder()

    # by default, Adasum doesn't need scaling up learning rate
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # move model to GPU.
        model.cuda()
        # if using GPU Adasum allreduce, scale learning rate by local_size
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler
    loss_function = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
    #                      momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.003)

    # Horovod: broadcast parameters & optimizer state
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer,
                                named_parameters=model.named_parameters(),
                                op=hvd.Average,
                                gradient_predivide_factor=args.gradient_predivide_factor)

    if hvd.rank()==0 and hvd.local_rank()==0:
        print('TIMER: broadcast:', timer()-st, 's') 

    et = timer()
    for epoch in range(args.epochs):
        lt = timer()
        loss_acc = 0.0
        count=0
        train_sampler.set_epoch(epoch)
        for data in train_loader:
            inputs = data[:-1][0]
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0)
            # ===================forward=====================
            optimizer.zero_grad()
            predictions = model(inputs.cuda())         # Forward pass
            loss = loss_function(predictions.float(), inputs.float().cuda())   # Compute loss function
            # ===================backward====================
            loss.backward()                                        # Backward pass
            optimizer.step()                                       # Optimizer step
            loss_acc+= loss.item()
            if count % args.log_int == 0 and hvd.rank()==0 and hvd.local_rank()==0:
                print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):.2f}% complete',\
                      f' / {timer() - lt:.2f} sec / train loss: {loss_acc} / loss: {loss.item():.6f}')
            count+=1

        if hvd.rank()==0 and hvd.local_rank()==0:
            print('TIMER: epoch time:', timer()-lt, 's') 

    if hvd.rank()==0 and hvd.local_rank()==0:
        print('TIMER: last epoch time:', timer()-lt, 's')
        print('TIMER: total epoch time:', timer()-et, 's')
        print('\n',torch.cuda.memory_summary(0),'\n')

    print('DEBUG: hvd.rank():',hvd.rank(),'hvd.local_rank():',hvd.local_rank(),
            ', torch.cuda.memory_reserved():',
            int(torch.cuda.memory_reserved(hvd.local_rank())/1024/1024),'MB')

    if hvd.rank()==0 and hvd.local_rank()==0:
        print('DEBUG: memory req:',
            int(torch.cuda.memory_reserved(hvd.local_rank())/1024/1024),'MB')

# start testing loop
    et = timer()
    model.eval()
    mean_sqr_diff = []
    count=0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[:-1][0]
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0)
            predictions = model(inputs.cuda())         # Forward pass
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            mean_sqr_diff.append(\
                torch.mean(torch.square(predictions.float()-inputs.float().cuda())).item())
            count+=1

    # mean from dataset (ignore if just 1 dataset)
    if count>1:
        mean_sqr_diff=np.mean(mean_sqr_diff)
    if hvd.rank()==0 and hvd.local_rank()==0:
        print(f'TIMER: total testing time: {timer()-et} s\n')

# finalise testing
    # mean from gpus
    avg_mean_sqr_diff = par_mean(mean_sqr_diff)
    avg_mean_sqr_diff = float(np.float_(avg_mean_sqr_diff.data.cpu().numpy()))
    if hvd.rank()==0 and hvd.local_rank()==0:
        print(f'DEBUG: avg_mean_sqr_diff: {avg_mean_sqr_diff}\n')

# clean-up
    if hvd.rank()==0 and hvd.local_rank()==0:
        print(f'TIMER: final time: {timer()-st} s\n')
    hvd.shutdown()

if __name__ == "__main__": 
    main()
    sys.exit()

# eof
