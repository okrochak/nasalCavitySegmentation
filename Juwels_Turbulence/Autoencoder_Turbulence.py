#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load ML libs 
import torch, torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import MNIST
from torchvision import datasets
from torch.nn.functional import interpolate
from torch.nn.functional import pad
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
    return torch.reshape(torch.cat((data_u, data_v, data_w)), (3,3,450,192)).permute((1,0,2,3))
    #return torch.reshape(torch.cat((data_u, data_v, data_w)), (3,3,192,192)).unfold(1, 1, 1).unfold(2,48,48).unfold(3,48,48).squeeze().permute(3,2,1,0,4,5).reshape(-1, 3, 48,48)

transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
turb_data = datasets.DatasetFolder('/p/project/prcoe12/RAISE/T31/Width1800', loader=hdf5_loader, transform=transform, extensions='.hdf5')
test_data = datasets.DatasetFolder('/p/project/prcoe12/RAISE/T31/testfolder', loader=hdf5_loader, transform=transform, extensions='.hdf5')

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
        # encoder - Convolutional layer is followed by a reLU activation which is then batch normalised
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
        
        # decoder - reLU activation is applied first followed by a batch normalisation and a convolutional layer
        # which maintains the same dimension and an interpolation layer in the end to decompress the dataset
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

    print('memory allocated:',torch.cuda.memory_reserved(device)/1024/1024, 'MB')
    
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
            # read the data to inputs, 0 tensor generated by the dataloader is eliminated
            inputs = data[:-1][0]
            # 3 layers closest to the wall are included in the batch dimension. This step increases
            # the batch size by a factor of 3
            inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0)
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
