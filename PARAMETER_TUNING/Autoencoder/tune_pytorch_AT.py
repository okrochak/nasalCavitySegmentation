#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI/RS/MA
# version: 220211a

# std libs
import argparse, sys, platform, os, time, numpy as np, h5py, random, shutil
import matplotlib.pyplot as plt
from functools import partial

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate, pad
from torchvision import datasets, transforms

from ray import train
import ray.train.torch
from ray import tune
from ray.tune import CLIReporter


# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    try:
        # small datase structure
        data_u = torch.from_numpy(np.array(f[list(f.keys())[0]]['u'])).permute((1,0,2))
        data_v = torch.from_numpy(np.array(f[list(f.keys())[0]]['v'])).permute((1,0,2))
        data_w = torch.from_numpy(np.array(f[list(f.keys())[0]]['w'])).permute((1,0,2))
    except:
        # large datase structure
        f1 = f[list(f.keys())[0]]
        data_u = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['u'])).permute((1,0,2))
        data_v = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['v'])).permute((1,0,2))
        data_w = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['w'])).permute((1,0,2))

    return torch.reshape(torch.cat((data_u, data_v, data_w)),
               (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3))

# parsed settings
def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='PyTorch actuated TBL')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')

    # model parsers
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wdecay', type=float, default=0.003,
                        help='weight decay in Adam optimizer (default: 0.003)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma in schedular (default: 0.95)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')
    parser.add_argument('--schedule', action='store_true', default=False,
                        help='enable scheduler in the training (default: False)')

    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training')

    # parallel parsers
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--nworker', type=int, default=0,
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs

# network
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

# train loop
def train(model, loss_function, train_loader, \
        optimizer, scheduler, epoch):
    loss_acc = 0.0
    for data in train_loader:
        inputs = data[:-1][0]
        inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0)
        # ===================forward=====================
        optimizer.zero_grad()
        predictions = model(inputs.float())         # Forward pass
        loss = loss_function(predictions.float(), inputs.float())   # Compute loss function
        # ===================backward====================
        loss.backward()                                        # Backward pass
        optimizer.step()                                       # Optimizer step
        loss_acc+= loss.item()
        scheduler.step()

    # profiling statistics

    loss_acc = par_mean(loss_acc, dist.get_world_size())
    
    return loss_acc

# test loop
def test(model, loss_function, test_loader):
    model.eval()
    test_loss = 0.0
    mean_sqr_diff = []
    count=0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[:-1][0]
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0)
            predictions = model(inputs.float())
            loss = loss_function(predictions.float(), inputs.float())
            test_loss+= loss.item()/inputs.shape[0]
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            mean_sqr_diff.append(\
                torch.mean(torch.square(predictions.float()-inputs.float())).item())
            count+=1

    # mean from dataset (ignore if just 1 dataset)
    if count>1:
        mean_sqr_diff=np.mean(mean_sqr_diff)

    # mean from gpus
    avg_test_loss = par_mean(test_loss, dist.get_world_size())
    avg_test_loss = float(np.float_(avg_test_loss.data.cpu().numpy()))
    avg_mean_sqr_diff = par_mean(mean_sqr_diff, dist.get_world_size())
    avg_mean_sqr_diff = float(np.float_(avg_mean_sqr_diff.data.cpu().numpy()))

    return avg_mean_sqr_diff

# PARALLEL HELPERS
# sum of field over GPGPUs
def par_sum(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    return res

# mean of field over GPGPUs
def par_mean(field,gwsize):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    res/=gwsize
    return res

# max(field) over GPGPUs
def par_max(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.MAX,group=None,async_op=True).wait()
    return res

# min(field) over GPGPUs
def par_min(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
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
    res = torch.Tensor([field])
    res = res.cuda() if args.cuda else res.cpu()
    dist.reduce(res,dst=dest,op=oper,group=None,async_op=False)
    return res

# gathers tensors from the whole group in a list (to all workers)
def par_allgather(field,gwsize):
    if args.cuda:
        sen = torch.Tensor([field]).cuda()
        res = [torch.Tensor([field]).cuda() for i in range(gwsize)]
    else:
        sen = torch.Tensor([field])
        res = [torch.Tensor([field]) for i in range(gwsize)]
    dist.all_gather(res,sen,group=None)
    return res

# gathers any object from the whole group in a list (to all workers)
def par_allgather_obj(obj,gwsize):
    res = [None]*gwsize
    dist.all_gather_object(res,obj,group=None)
    return res
     
def train_func(config):
    
    # load datasets
        transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        turb_data = datasets.DatasetFolder(config["data_dir"]+'trainfolder',\
            loader=hdf5_loader, transform=transform, extensions='.hdf5')
        test_data = datasets.DatasetFolder(config["data_dir"]+'testfolder',\
             loader=hdf5_loader, transform=transform, extensions='.hdf5')


        train_loader = torch.utils.data.DataLoader(turb_data, batch_size=config["batch_size"], num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"], num_workers=4, pin_memory=True)

        train_loader = ray.train.torch.prepare_data_loader(train_loader)
        test_loader = ray.train.torch.prepare_data_loader(test_loader)

        # create model
        model = autoencoder()

        distrib_model = ray.train.torch.prepare_model(model)

    # optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(distrib_model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["gamma"])

    # start trainin loop
        for epoch in range(config["epochs"]):
            loss_acc = train(distrib_model, loss_function, train_loader, optimizer, scheduler_lr, epoch)
            

    # start testing loop
            avg_mean_sqr_diff = test(distrib_model, loss_function, test_loader)
        
            #print("Epoch: {} Train Loss: {} Test Loss: {}".format(epoch, loss_acc, avg_mean_sqr_diff))
            
            ray.train.report(train_loss= loss_acc, test_loss=avg_mean_sqr_diff)
        
# MAIN
#
#
def main(num_samples=20, gpus_per_trial=4):
    # get parse args
    pars_ini()
    
    ray.init(address='auto', _temp_dir=os.path.abspath(os.getcwd()))

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # get directory
    program_dir = os.getcwd()
    
    
    # create a Ray Train trainer
    from ray.train import Trainer
    trainer = Trainer(backend="torch", num_workers=gpus_per_trial, use_gpu=True)
    
    # convert the train function to a Ray trainable
    trainable = trainer.to_tune_trainable(train_func)
    
    
    # restrict Ray Tune output to once every 2 mins
    reporter = CLIReporter(
        metric_columns=["train_loss", "test_loss", "training_iteration", "time_total_s"],
        max_report_frequency=120)

    # define the search space
    config = {"batch_size": tune.choice([8, 16, 32, 64]), 
        "lr": tune.choice([0.1, 0.001, 0.0001]),
        "wd": args.wdecay,
        "gamma": args.gamma,
        "data_dir": args.data_dir,
        "epochs": args.epochs
        }

    # launch tuning run
    result = tune.run(
        trainable,
        local_dir=os.path.join(os.path.abspath(os.getcwd()), "ray_results"),
        config=config,
        num_samples=num_samples,
        verbose=1,
        progress_reporter=reporter,
        scheduler=None)
    
    
    # save results to json file
    df = result.results_df
    
    df.to_json('AT_tuning_results.json', default_handler=str)

if __name__ == "__main__":
    main(num_samples=20, gpus_per_trial=4)
    sys.exit()

# eof
