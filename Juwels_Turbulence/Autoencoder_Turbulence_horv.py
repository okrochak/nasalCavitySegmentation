#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load ML libs 
import torch, torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import MNIST
from torchvision import datasets
from torch.nn.functional import interpolate

import horovod.torch as hvd


# load system libs 
import os, struct, numpy as np, sys, platform, argparse, h5py
from timeit import default_timer as timer

# some debug
print('sys ver:',sys.version)
print('python ver:',platform.python_version())
print('cuda:',torch.cuda.is_available())
print('cuda current device:',torch.cuda.current_device())
print('cuda device count:',torch.cuda.device_count())
print('GPU:',torch.cuda.get_device_name(0))
print('GPU address',torch.cuda.get_device_properties)

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    data_u = np.asarray(f[list(f.keys())[0]]['u']).transpose((1,0,2))
    data_v = np.asarray(f[list(f.keys())[0]]['v']).transpose((1,0,2))
    data_w = np.asarray(f[list(f.keys())[0]]['w']).transpose((1,0,2))
    return np.concatenate((data_u, data_v, data_w))

turb_data = datasets.DatasetFolder('/p/project/prcoe12/RAISE/T31', loader=hdf5_loader, extensions='.hdf5')

# Class defining the CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 1 class
        self.out = nn.Linear(32 * 7 * 7, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.reLU = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.tanhf = nn.Tanh()
        #Encoding layers
        self.conv_en1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_en2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_en3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        #Decoding layers
        self.conv_de1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_de3 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # encoder
        x = self.conv_en1(x)
        out1 = self.leaky_reLU(x)
        x = self.pool(x)
        x = self.conv_en2(x)
        out2 = self.leaky_reLU(x)
        x = self.pool(x)
        x = self.conv_en3(x)
        out3 = self.leaky_reLU(x)
        x = self.pool(x)
        # decoder
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_de1(x)
        x = self.leaky_reLU(x)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_de2(x)
        x = self.leaky_reLU(x)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_de3(x)
        x = self.tanhf(x)
        return x

def main():
    # get arguments
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument('--batch_size',type=int,default=5,action='store',metavar='N',help='batch size')
    parser.add_argument('--epochs',type=int,default=1,action='store',metavar='N',help='#epochs')
    parser.add_argument('--learning_rate',type=float,default=0.001,action='store',metavar='N',help='learing rate')
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    learning_rate = args.learning_rate
    print('batch size: ',BATCH_SIZE)
    print('#epochs: ',NUM_EPOCHS)
    print('learning rate: ',learning_rate)
    
    #args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: initialize library.
    hvd.init()

    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
        
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)
    
    #kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # get evn
    #local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    #print('local_rank: ',local_rank)

    #n_nodes = int(os.environ['SLURM_NNODES'])
    #ngpus_per_node = torch.cuda.device_count()

    # initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # for single node multi gpu
    #torch.distributed.init_process_group(backend='gloo', init_method='env://')
    # for multi node multi gpu
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #print('initialised')

    # encapsulate the model on the GPU assigned to the current process
    #device = torch.device('cuda', local_rank)
    #torch.cuda.set_device(local_rank)
    #print('device set')

    # Have all workers wait - prints errors at the moment
    # torch.distributed.barrier()

    # dataset 
    transform = transforms.Compose([transforms.ToTensor()])
    # restricts data loading to a subset of the dataset exclusive to the current process
    sampler = torch.utils.data.distributed.DistributedSampler(turb_data, num_replicas=hvd.size(), rank=hvd.rank())
    
    # load data to GPUs
    dataloader = torch.utils.data.DataLoader(dataset=turb_data, sampler=sampler, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False)
    print('data loaded')
    #print(len(dataloader.dataset))
    # model distribute
    #model = autoencoder().to(device)
    #distrib_model = torch.nn.parallel.DistributedDataParallel(model, \
	#	    device_ids=[device], output_device=device)
    #print('model distributed')
    
    model = autoencoder()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size()

    # Move model to GPU.
    model.cuda()
 
    # some stuff
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder().parameters(), lr=learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    start = timer()
    for epoch in range(NUM_EPOCHS):
        loss_acc = 0.0
        count = 0
        for data in dataloader:
            inputs = torch.stack((data[0][0],data[0][1],data[0][2]))
            inputs.cuda()
            #inputs = torch.transpose(inputs,0,1)
            # ===================forward=====================
            optimizer.zero_grad()
            predictions = model(inputs.cuda())         # Forward pass
            # ===================backward====================
            loss = loss_function(predictions, inputs.cuda())
            loss.backward()                                        # Backward pass
            optimizer.step()                                       # Optimizer step
            
            loss_acc+= loss.item()
            count+= 1
            print(count)
            
            if count%1000==0:
                print(f'Epoch: {epoch}\t{100 * (count + 1) / len(dataloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.', end='\r')
    total_time = timer()-start    
    #print(predictions, labels)
    print(f'Training time: {total_time}\n')
if __name__ == "__main__": 
    main()
