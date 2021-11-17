#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:42:43 2021

@author: rakeshsarma
"""

#load ML libs
import torch, torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision import datasets
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

# load system libs
import os, struct, numpy as np, sys, platform, argparse, h5py
from timeit import default_timer as timer

#SEED = 42
BATCH_SIZE = 100
NUM_EPOCHS = 10
print(BATCH_SIZE)

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    data_u = torch.FloatTensor(f[list(f.keys())[0]]['u']).permute((1,0,2))
    data_v = torch.FloatTensor(f[list(f.keys())[0]]['v']).permute((1,0,2))
    data_w = torch.FloatTensor(f[list(f.keys())[0]]['w']).permute((1,0,2))
    return torch.cat((data_u, data_v, data_w))

transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))])
turb_data = datasets.DatasetFolder('/p/project/prcoe12/RAISE/T31', loader=hdf5_loader, transform=transform, extensions='.hdf5')

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

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
        x = self.leaky_reLU(x)
        x = self.pool(x)
        x = self.conv_en2(x)
        x = self.leaky_reLU(x)
        x = self.pool(x)
        x = self.conv_en3(x)
        x = self.leaky_reLU(x)
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
    
model = autoencoder()
model.to(device)

learning_rate = 0.0001
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(learning_rate)
print(device)
dataloader = DataLoader(dataset=turb_data, batch_size=BATCH_SIZE)


for epoch in range(NUM_EPOCHS):
    loss_acc = 0.0
    count = 0
    start = timer()
    for data in dataloader:
        # ===================forward=====================
        optimizer.zero_grad()
        predictions = model(data[:-1][0].to(device))         # Forward pass
        loss = loss_function(predictions.float(), data[:-1][0].float().to(device))   # Compute loss function
        # ===================backward====================
        loss.backward()                                        # Backward pass
        optimizer.step()                                       # Optimizer step
        
        loss_acc+= loss.item()
        count+= 1
 
        if count%10==0:
            print(f'Epoch: {epoch}\t{100 * (count + 1) / len(dataloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.', end='\r')
    print(f'Epoch: {epoch},\t Loss: {loss_acc}\n')
print(predictions, data[:-1][0])
