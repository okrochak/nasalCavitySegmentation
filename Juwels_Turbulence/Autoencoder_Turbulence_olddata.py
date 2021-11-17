#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load ML libs 
import torch, torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import MNIST
from torchvision import datasets
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
# load system libs
import os, struct, numpy as np, sys, platform, argparse, h5py
from timeit import default_timer as timer
import random

# some debug
print('sys ver:',sys.version)
print('python ver:',platform.python_version())
print('cuda:',torch.cuda.is_available())
print('cuda current device:',torch.cuda.current_device())
print('cuda device count:',torch.cuda.device_count())
print('GPU:',torch.cuda.get_device_name(0))
print('GPU address',torch.cuda.get_device_properties)


def plot_scatter(inp_img, out_img):
    fig = plt.figure(figsize = (4,8))
    plt.rcParams.update({'font.size': 10})
    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(inp_img, vmin = np.min(inp_img), vmax = np.max(inp_img), interpolation='None')
    ax1.set_title('Input')
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im1, cax=cax, orientation='vertical')
    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(out_img, vmin = np.min(inp_img), vmax = np.max(inp_img), interpolation='None')
    ax2.set_title('Output')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.99, 0.396, 0.03, 0.225])
    fig.tight_layout(pad=1.0)
    fig.colorbar(im1, cax=cbar_ax)
    plt.savefig('vfield_recon_CAE'+str(random.randint(0,100))+'.pdf', bbox_inches = 'tight', pad_inches = 0)

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    data_u = torch.FloatTensor(f[list(f.keys())[0]]['u']).permute((1,0,2))
    data_v = torch.FloatTensor(f[list(f.keys())[0]]['v']).permute((1,0,2))
    data_w = torch.FloatTensor(f[list(f.keys())[0]]['w']).permute((1,0,2))
    return torch.reshape(torch.cat((data_u, data_v, data_w)), (3,3,192,192)).permute((1,0,2,3))

transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
turb_data = datasets.DatasetFolder('/p/project/prcoe12/RAISE/T31/old.Width1800', loader=hdf5_loader, transform=transform, extensions='.hdf5')
test_data = datasets.DatasetFolder('/p/project/prcoe12/RAISE/T31/old.testfolder', loader=hdf5_loader, transform=transform, extensions='.hdf5')

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.reLU = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.tanhf = nn.Tanh()
        #Encoding layers
        self.conv_en1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bnorm_en1 = nn.BatchNorm2d(16)
        self.conv_en2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bnorm_en2 = nn.BatchNorm2d(32)
        #self.conv_en3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        #self.bnorm_en3 = nn.BatchNorm2d(64)
        #self.conv_en4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        #Decoding layers
        #self.conv_de1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        #self.conv_de2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.bnorm_de2 = nn.BatchNorm2d(64)
        #self.bnorm_de3 = nn.BatchNorm2d(64)
        #self.conv_de3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm_de4 = nn.BatchNorm2d(32)
        self.conv_de4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bnorm_de5 = nn.BatchNorm2d(16)
        self.conv_de5 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # encoder
        x = self.conv_en1(x)
        x = self.reLU(x)
        x = self.bnorm_en1(x)
        x = self.conv_en2(x)
        x = self.reLU(x)
        x = self.bnorm_en2(x)
        #x = self.conv_en3(x)
        #x = self.reLU(x)
        #x = self.bnorm_en3(x)
        #x = self.pool(x)
        #x = self.conv_en4(x)
        #x = self.leaky_reLU(x)
        # decoder
        #x = self.conv_de1(x)
        #x = self.leaky_reLU(x)
        #x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #x = self.conv_de2(x)
        #x = self.reLU(x)
        #x = self.bnorm_de3(x)
        #x = self.conv_de3(x)
        #x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.reLU(x)
        x = self.bnorm_de4(x)
        x = self.conv_de4(x)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
       	x = self.reLU(x)
        x = self.bnorm_de5(x)
        x = self.conv_de5(x)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
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

    # get evn
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    print('local_rank: ',local_rank)

    n_nodes = int(os.environ['SLURM_NNODES'])
    ngpus_per_node = torch.cuda.device_count()

    # initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # for single node multi gpu
    #torch.distributed.init_process_group(backend='gloo', init_method='env://')
    # for multi node multi gpu
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print('initialised')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(local_rank)
    print('device set')

    # Have all workers wait - prints errors at the moment
    # torch.distributed.barrier()

    # restricts data loading to a subset of the dataset exclusive to the current process
    sampler = torch.utils.data.distributed.DistributedSampler(turb_data, num_replicas=n_nodes*ngpus_per_node)
    sampler_test = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=n_nodes*ngpus_per_node)
    #sampler = torch.utils.data.distributed.DistributedSampler(turb_data)
    # load data to GPUs
    dataloader = torch.utils.data.DataLoader(dataset=turb_data, sampler=sampler, num_workers=0, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset=test_data, sampler=sampler_test, num_workers=0, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False)
    print('data loaded')
    #print(len(dataloader.dataset))
    # model distribute
    model = autoencoder().to(device)
    distrib_model = torch.nn.parallel.DistributedDataParallel(model, \
		    device_ids=[device], output_device=device)
    print('model distributed')

    # some stuff
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.003)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print('memory allocated:',torch.cuda.memory_allocated(device)/1024/1024, 'MB')
    
    loss_perc_mean = []
    loss_perc_min = []
    loss_perc_max = []
    loss_perc_std = []
    start = timer()
    for epoch in range(NUM_EPOCHS):
        loss_acc = 0.0
        count = 0
        sampler.set_epoch(epoch)
        torch.distributed.barrier()
        for data in dataloader:
            inputs = data[:-1][0]
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0)
	    #inputs = torch.transpose(inputs,0,1)
            # ===================forward=====================
            optimizer.zero_grad()
            predictions = distrib_model(inputs.to(device))         # Forward pass
            loss = loss_function(predictions.float(), inputs.float().to(device))   # Compute loss function
            # ===================backward====================
            loss.backward()                                        # Backward pass
            optimizer.step()                                       # Optimizer step
            #print(loss) 
            loss_acc+= loss.item()/inputs.shape[0]
            count+= 1
            #print(torch.mean((torch.abs(predictions-inputs.to(device))/torch.abs(inputs.to(device)))))
            if count%1000==0:
                print(f'Epoch: {epoch}\t{100 * (count + 1) / len(dataloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.', end='\r')
            if epoch==NUM_EPOCHS-1:
                loss_perc_mean.append(torch.mean((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
                loss_perc_max.append(torch.max((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
                loss_perc_min.append(torch.min((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
                loss_perc_std.append(torch.std((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
        scheduler_lr.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print(f'Epoch: {epoch}\t Training loss: {loss_acc}\n')
        total_time = timer()-start
    #print(predictions, inputs)
    #print(f'Loss_mean: {loss_perc_mean}\n')
    #print(f'Loss_max: {loss_perc_max}\n')
    #print(f'Loss_min: {loss_perc_min}\n')
    #print(f'Loss_std: {loss_perc_std}\n')
    #print(f'Training time: {total_time}\n')
    #plot_scatter(inputs[0][0].cpu().detach().numpy(), predictions[0][0].cpu().detach().numpy())
    
    #test section starts
    distrib_model.eval()
    test_loss = 0.0
    loss_perc_mean = []
    loss_perc_max = []
    loss_perc_min = []
    loss_perc_std = []
    with torch.no_grad():
        for data in dataloader_test:
            inputs = data[:-1][0]
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0)
            predictions = distrib_model(inputs.to(device))
            loss = loss_function(predictions.float(), inputs.float().to(device))
            test_loss+= loss.item()/inputs.shape[0]
            #loss_perc_mean.append(torch.mean((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
            #loss_perc_max.append(torch.max((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
            #loss_perc_min.append(torch.min((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
            #loss_perc_std.append(torch.std((torch.abs((predictions.float()-inputs.float().to(device))/inputs.float().to(device)))).item())
    print(f'Test loss: {test_loss}\n')
    #print(f'Loss_mean: {loss_perc_mean}\n')
    #print(f'Loss_max: {loss_perc_max}\n')
    #print(f'Loss_min: {loss_perc_min}\n')
    #print(f'Loss_std: {loss_perc_std}\n')
    plot_scatter(inputs[0][0].cpu().detach().numpy(), predictions[0][0].cpu().detach().numpy())


if __name__ == "__main__": 
    main()
