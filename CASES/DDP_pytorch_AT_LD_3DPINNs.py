#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: RS
# version: 211029a

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
        
# plot reconstruction
def plot_scatter(inp_img, out_img, data_org):
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
    plt.savefig('vfield_recon_VAE'+data_org+str(random.randint(0,100))+'.pdf',
                    bbox_inches = 'tight', pad_inches = 0)

def uniform_data(data):
    if data.shape[2]==300 or data.shape[2]==400 or data.shape[2]==450:    
        data_out = data.reshape(data.shape[0],data.shape[1],data.shape[2]//50,50,data.shape[3])
        data_out = torch.cat((data_out[:,:,:5,:,:].reshape(data.shape[0],data.shape[1],250,data.shape[3]),
                            data_out[:,:,-5:,:,:].reshape(data.shape[0],data.shape[1],250,data.shape[3])
                            ))
    elif data.shape[2]==750: 
        data_out = data.reshape(data.shape[0],data.shape[1],data.shape[2]//50,50,data.shape[3])
        data_out = torch.cat((data_out[:,:,:5,:,:].reshape(data.shape[0],data.shape[1],250,data.shape[3]),
                            data_out[:,:,5:10,:,:].reshape(data.shape[0],data.shape[1],250,data.shape[3]),
                            data_out[:,:,10:,:,:].reshape(data.shape[0],data.shape[1],250,data.shape[3])
                            ))
    else:
        data_out=data
    return data_out

def collate_batch(batch):
    imgs = [item[0].view(int(item[0].shape[0]/89), 89, item[0].shape[1], \
        item[0].shape[2], item[0].shape[3]) for item in batch]
    targets = [item[1] for item in batch]
    imgs = torch.cat((imgs))
    return imgs, targets

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    #try:
        # small datase structure
    #    data_u = torch.from_numpy(np.array(f[list(f.keys())[0]]['u'])).permute((1,0,2))[:-9,:,:] # removes last ten layers assuming free stream
    #    data_u = 2*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1
    #    data_v = torch.from_numpy(np.array(f[list(f.keys())[0]]['v'])).permute((1,0,2))[:-9,:,:]
    #    data_v = 2*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1
    #    data_w = torch.from_numpy(np.array(f[list(f.keys())[0]]['w'])).permute((1,0,2))[:-9,:,:]
    #    data_w = 2*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1
    #except:
        # large datase structure
    f1 = f[list(f.keys())[0]]
    data_u = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['u'])).permute((1,0,2))[:-9,:,:]
    data_u = 2*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1
    data_v = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['v'])).permute((1,0,2))[:-9,:,:]
    data_v = 2*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1
    data_w = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['w'])).permute((1,0,2))[:-9,:,:]
    data_w = 2*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1
    return uniform_data(torch.reshape(torch.cat((data_u, data_v, data_w)),
               (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3)))

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
    parser.add_argument('--log-int', type=int, default=30,
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
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.unpool = nn.MaxUnpool3d(kernel_size=2, stride=2, padding=0)
        
#Encoding layers - conv_en1 denotes 1st (1) convolutional layer for the encoding (en) part
        
        # Encoder
        self.cae_conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.cae_bn1 = nn.BatchNorm3d(16)
        self.cae_conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.cae_bn2 = nn.BatchNorm3d(32)
        self.cae_conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cae_bn3 = nn.BatchNorm3d(64)
        self.cae_conv4 = nn.Conv3d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.cae_bn4 = nn.BatchNorm3d(16)

        #Decoder
        self.cae_conv5 = nn.Conv3d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cae_bn5 = nn.BatchNorm3d(64)
        self.cae_conv6 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.cae_bn6 = nn.BatchNorm3d(32)
        self.cae_conv7 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=[0,1,1], bias=False)
        self.cae_bn7 = nn.BatchNorm3d(16)
        self.cae_conv8 = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=0, bias=False)

    def forward(self, inp_x):
        """
        encoder - Convolutional layer is followed by a reLU activation which is then batch normalised
        """
        # encoding
        cae_conv1 = self.leaky_reLU(self.cae_bn1(self.cae_conv1(inp_x)))
        cae_conv2 = self.leaky_reLU(self.cae_bn2(self.cae_conv2(cae_conv1)))
        cae_conv3 = self.leaky_reLU(self.cae_bn3(self.cae_conv3(cae_conv2)))
        cae_conv4 = self.leaky_reLU(self.cae_bn4(self.cae_conv4(cae_conv3)))
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
# 1st decoding layer
        # decoding
        cae_conv5 = self.cae_bn5(self.cae_conv5(cae_conv4))
        cae_conv5 = interpolate(cae_conv5, scale_factor=2, mode='trilinear', align_corners=True)
        cae_conv5 = self.leaky_reLU(cae_conv5)

        cae_conv6 = self.leaky_reLU(self.cae_bn6(self.cae_conv6(cae_conv5)))

        cae_conv7 = self.cae_bn7(self.cae_conv7(cae_conv6))
        cae_conv7 = interpolate(cae_conv7, scale_factor=2, mode='trilinear', align_corners=True)
        cae_conv7 = pad(cae_conv7,(1,1,0,0,1,2))
        cae_conv7 = self.leaky_reLU(cae_conv7)

        out_x = self.cae_conv8(cae_conv7)
        return out_x

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

# deterministic dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# train loop
def train(model, sampler, loss_function, device, train_loader, optimizer, epoch, grank, lt, scheduler_lr, del_x, coords_y, del_z):
    loss_acc = 0.0
    count=0
    weights_y = []
    #for i in range(89-1):
    #        weights_y.append(torch.tensor([[-del_y[i+1]/(del_y[i]*(del_y[i]+del_y[i+1])),
    #            (del_y[i+1]-del_y[i])/(del_y[i+1]*del_y[i]),del_y[i]/(del_y[i+1]*(del_y[i]+del_y[i+1]))],[-del_y[i+1]/(del_y[i]*(del_y[i]+del_y[i+1])),
    #            (del_y[i+1]-del_y[i])/(del_y[i+1]*del_y[i]),del_y[i]/(del_y[i+1]*(del_y[i]+del_y[i+1]))],[-del_y[i+1]/(del_y[i]*(del_y[i]+del_y[i+1])),
    #            (del_y[i+1]-del_y[i])/(del_y[i+1]*del_y[i]),del_y[i]/(del_y[i+1]*(del_y[i]+del_y[i+1]))]]))
    #sampler.set_epoch(epoch)
    for inputs in train_loader:
        inps = inputs[0].permute(0,2,1,3,4).squeeze(0).float().to(device)
        # ===================forward=====================
        optimizer.zero_grad()
        predictions = model(inps)         # Forward pass
        # =========== for incompressibility constraint =============
        #weights_x = torch.tensor([[0.,0.,0.],[-1/(2*del_x),0.,1/(2*del_x)],[0.,0.,0.]])
        #preds_sh = predictions.shape
        #weights_x = weights_x.view(1, 1, 3, 3).repeat(preds_sh[1]*preds_sh[2],1,1,1).to(device)
        #weights_x.requires_grad=True
        #dphi_dx = torch.nn.functional.conv2d(predictions.reshape(preds_sh[0],preds_sh[1]*\
        #        preds_sh[2],preds_sh[3], preds_sh[4]), weights_x, groups=preds_sh[1]*\
        #        preds_sh[2]).reshape(preds_sh[0],preds_sh[1],preds_sh[2],preds_sh[3]-2, \
        #        preds_sh[4]-2)
        #dphi_dx = dphi_dx[:,:,1:-1,:,:]
        
        dphi_dx = torch.gradient(predictions, spacing=del_x, dim=[4], edge_order=2)[0]
        dphi_dy = torch.gradient(predictions, spacing=(torch.tensor(coords_y[:-9]).to(device),), dim=[2], edge_order=2)[0]
        dphi_dz = torch.gradient(predictions, spacing=del_z, dim=[3], edge_order=2)[0]

        #weights_z = torch.tensor([[0.,0.,0.],[-1/(2*del_z),0.,1/(2*del_z)],[0.,0.,0.]])
        #weights_z = weights_z.view(1, 1, 3, 3).repeat(preds_sh[1]*preds_sh[2],1,1,1).to(device)
        #weights_z.requires_grad=True
        #dphi_dz = torch.nn.functional.conv2d(predictions.permute(0,1,2,4,3).reshape(preds_sh[0],preds_sh[1]*\
        #        preds_sh[2],preds_sh[4], preds_sh[3]), weights_z, groups=preds_sh[1]*\
        #        preds_sh[2]).reshape(preds_sh[0],preds_sh[1],preds_sh[2],preds_sh[4]-2, \
        #        preds_sh[3]-2).permute(0,1,2,4,3)
        #dphi_dz = dphi_dz[:,:,1:-1,:,:]

        #predictions = predictions.reshape(preds_sh[0],preds_sh[1],preds_sh[2],preds_sh[3],preds_sh[4]).permute(0,1,3,4,2)

        #dphi_dy = torch.empty(preds_sh[0],preds_sh[1],preds_sh[3],preds_sh[4]-2,preds_sh[2]-2).to(device)
        #for j in range(preds_sh[3]):
        #    for k in range(preds_sh[4]-2):
        #        for i in range(preds_sh[2]-2):
        #            weights_y[i] = weights_y[i].to(device)
        #            weights_y[i].requires_grad=True
        #            dphi_dy[:,:,j,k,i] = torch.sum(torch.mul(predictions[:,:,j,k:k+3,i:i+3], weights_y[i]),dim=[2,3])
        #dphi_dy = dphi_dy.permute(0,1,4,2,3)[:,:,:,1:-1,:]        


        #predictions_pinns = torch.zeros((preds_sh[0]*preds_sh[1],preds_sh[2],preds_sh[3],preds_sh[4])).to(device)
        #predictions_pinns[torch.arange(0,preds_sh[0]*3,3),:,:,:] = (dphi_dy[:,2,:,:,:]-dphi_dz[:,1,:,:,:])
        #predictions_pinns[torch.arange(1,preds_sh[0]*3,3),:,:,:] = (dphi_dz[:,0,:,:,:]-dphi_dx[:,2,:,:,:])
        #predictions_pinns[torch.arange(2,preds_sh[0]*3,3),:,:,:] = (dphi_dx[:,1,:,:,:]-dphi_dy[:,0,:,:,:])

        #loss = loss_function(predictions_pinns.float(), inps.reshape(preds_sh[0]*preds_sh[1],preds_sh[2],preds_sh[3],preds_sh[4])[:,1:-1,1:-1,1:-1].float().to(device))   # Compute loss function

        predictions_pinns = torch.zeros((predictions.shape[0],predictions.shape[1],predictions.shape[2],predictions.shape[3],predictions.shape[4])).to(device)

        predictions_pinns[:,0,:,:,:] = dphi_dy[:,2,:,:,:]-dphi_dz[:,1,:,:,:]
        predictions_pinns[:,1,:,:,:] = dphi_dz[:,0,:,:,:]-dphi_dx[:,2,:,:,:]
        predictions_pinns[:,2,:,:,:] = dphi_dx[:,1,:,:,:]-dphi_dy[:,0,:,:,:]

        loss = loss_function(predictions_pinns.float(), inps)

        # ===================backward====================
        loss.backward()                                        # Backward pass
        optimizer.step()                                       # Optimizer step
        loss_acc+= loss.item()
        if count % args.log_int == 0 and grank==0:
            print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):3.2f}% complete',\
                  f' / {time.time() - lt:.2f} s / accumulated loss: {loss_acc}')
        count+=1

    if args.schedule:
        scheduler_lr.step()

    # profiling statistics
    if grank==0:
        print('TIMER: epoch time:', time.time()-lt, 's\n')

    return loss_acc

# test loop
def test(model, loss_function, device, test_loader, grank, gwsize, del_x, coords_y, del_z):
    et = time.time()
    model.eval()
    test_loss = 0.0
    mean_sqr_diff = []
    count=0
    with torch.no_grad():
        for inputs in test_loader:
            inps = inputs[0].permute(0,2,1,3,4).squeeze(0).float().to(device)
            predictions = model(inps)

            dphi_dx = torch.gradient(predictions, spacing=del_x, dim=[4], edge_order=2)[0]
            dphi_dy = torch.gradient(predictions, spacing=(torch.tensor(coords_y[:-9]).to(device),), dim=[2], edge_order=2)[0]
            dphi_dz = torch.gradient(predictions, spacing=del_z, dim=[3], edge_order=2)[0]

            #weights_x = torch.tensor([[0.,0.,0.],[-1/(2*del_x),0.,1/(2*del_x)],[0.,0.,0.]])
            #preds_sh = predictions.shape
            #weights_x = weights_x.view(1, 1, 3, 3).repeat(preds_sh[1]*preds_sh[2],1,1,1).to(device)
            #weights_x.requires_grad=True
            #dphi_dx = torch.nn.functional.conv2d(predictions.reshape(preds_sh[0],preds_sh[1]*\
            #    preds_sh[2],preds_sh[3], preds_sh[4]), weights_x, groups=preds_sh[1]*\
            #    preds_sh[2]).reshape(preds_sh[0],preds_sh[1],preds_sh[2],preds_sh[3]-2, \
            #    preds_sh[4]-2)
            #dphi_dx = dphi_dx[:,:,1:-1,:,:]

            #weights_z = torch.tensor([[0.,0.,0.],[-1/(2*del_z),0.,1/(2*del_z)],[0.,0.,0.]])
            #weights_z = weights_z.view(1, 1, 3, 3).repeat(preds_sh[1]*preds_sh[2],1,1,1).to(device)
            #weights_z.requires_grad=True
            #dphi_dz = torch.nn.functional.conv2d(predictions.permute(0,1,2,4,3).reshape(preds_sh[0],preds_sh[1]*\
            #    preds_sh[2],preds_sh[4], preds_sh[3]), weights_z, groups=preds_sh[1]*\
            #    preds_sh[2]).reshape(preds_sh[0],preds_sh[1],preds_sh[2],preds_sh[4]-2, \
            #    preds_sh[3]-2).permute(0,1,2,4,3)
            #dphi_dz = dphi_dz[:,:,1:-1,:,:]

            #predictions = predictions.reshape(preds_sh[0],preds_sh[1],preds_sh[2],preds_sh[3],preds_sh[4]).permute(0,1,3,4,2)
            #weights_y = []
            #for i in range(preds_sh[2]-1):
            #    weights_y.append(torch.tensor([[-del_y[i+1]/(del_y[i]*(del_y[i]+del_y[i+1])),
            #        (del_y[i+1]-del_y[i])/(del_y[i+1]*del_y[i]),del_y[i]/(del_y[i+1]*(del_y[i]+del_y[i+1]))],[-del_y[i+1]/(del_y[i]*(del_y[i]+del_y[i+1])),
            #        (del_y[i+1]-del_y[i])/(del_y[i+1]*del_y[i]),del_y[i]/(del_y[i+1]*(del_y[i]+del_y[i+1]))],[-del_y[i+1]/(del_y[i]*(del_y[i]+del_y[i+1])),
            #        (del_y[i+1]-del_y[i])/(del_y[i+1]*del_y[i]),del_y[i]/(del_y[i+1]*(del_y[i]+del_y[i+1]))]]))

            #dphi_dy = torch.empty(preds_sh[0],preds_sh[1],preds_sh[3],preds_sh[4]-2,preds_sh[2]-2)
            #for j in range(preds_sh[3]):
            #    for k in range(preds_sh[4]-2):
            #        for i in range(preds_sh[2]-2):
            #            weights_y[i] = weights_y[i].to(device)
            #            weights_y[i].requires_grad=True
            #            dphi_dy[:,:,j,k,i] = torch.sum(torch.mul(predictions[:,:,j,k:k+3,i:i+3], weights_y[i]),dim=[2,3])
            #dphi_dy = dphi_dy.permute(0,1,4,2,3)[:,:,:,1:-1,:] 

            #predictions_pinns = torch.zeros((preds_sh[0]*preds_sh[1],preds_sh[2]-2,preds_sh[3]-2,preds_sh[4]-2)).to(device)
            #predictions_pinns[torch.arange(0,preds_sh[0]*3,3),:,:,:] = (dphi_dy[:,2,:,:,:]-dphi_dz[:,1,:,:,:])
            #predictions_pinns[torch.arange(1,preds_sh[0]*3,3),:,:,:] = (dphi_dz[:,0,:,:,:]-dphi_dx[:,2,:,:,:])
            #predictions_pinns[torch.arange(2,preds_sh[0]*3,3),:,:,:] = (dphi_dx[:,1,:,:,:]-dphi_dy[:,0,:,:,:])

            predictions_pinns = torch.zeros((predictions.shape[0],predictions.shape[1],predictions.shape[2],predictions.shape[3],predictions.shape[4])).to(device)

            predictions_pinns[:,0,:,:,:] = dphi_dy[:,2,:,:,:]-dphi_dz[:,1,:,:,:]
            predictions_pinns[:,1,:,:,:] = dphi_dz[:,0,:,:,:]-dphi_dx[:,2,:,:,:]
            predictions_pinns[:,2,:,:,:] = dphi_dx[:,1,:,:,:]-dphi_dy[:,0,:,:,:]

            #loss = loss_function(predictions_pinns.float(), inps.reshape(preds_sh[0]*preds_sh[1],preds_sh[2],preds_sh[3],preds_sh[4])[:,1:-1,1:-1,1:-1].float().to(device))   # Compute loss function
            loss = loss_function(predictions_pinns.float(), inps)

            test_loss+= loss.item()/inps.shape[0]
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            #mean_sqr_diff.append(\
            #    torch.mean(torch.square(predictions_pinns.float(), inps.reshape(preds_sh[0]*preds_sh[1],preds_sh[2],preds_sh[3],preds_sh[4])[:,1:-1,1:-1,1:-1].float())).item())
            mean_sqr_diff.append(torch.mean(torch.square(predictions_pinns.float()-inps.float())).item())
            count+=1

    # mean from dataset (ignore if just 1 dataset)
    if count>1:
        mean_sqr_diff=np.mean(mean_sqr_diff)

    if grank==0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time()-et} s')
        if not args.testrun or not args.benchrun:
            plot_scatter(inps[0][0][0].cpu().detach().numpy(), 
                    predictions_pinns[0][0][0].cpu().detach().numpy(), 'test')

    # mean from gpus
    avg_mean_sqr_diff = par_mean(mean_sqr_diff,gwsize)
    avg_mean_sqr_diff = float(np.float_(avg_mean_sqr_diff.data.cpu().numpy()))
    if grank==0:
        print(f'DEBUG: avg_mean_sqr_diff: {avg_mean_sqr_diff}\n')

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
#
#
# MAIN
#
#
def main():
    # get parse args
    pars_ini()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

# initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend=args.backend)

# deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    # get job rank info - rank==0 master gpu
    lwsize = torch.cuda.device_count() if args.cuda else 0 # local world size - per node
    gwsize = dist.get_world_size()     # global world size - per run
    grank = dist.get_rank()            # global rank - assign per run
    lrank = dist.get_rank()%lwsize     # local rank - assign per node

    # some debug
    if grank==0: 
        print('TIMER: initialise:', time.time()-st, 's')
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:',sys.version,'\n')

        print('DEBUG: IO parsers:')
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.restart_int:',args.restart_int,'\n')

        print('DEBUG: model parsers:')        
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.lr:',args.lr)
        print('DEBUG: args.wdecay:',args.wdecay)
        print('DEBUG: args.gamma:',args.gamma)
        print('DEBUG: args.schedule:',args.schedule)
        print('DEBUG: args.shuff:',args.shuff,'\n')

        print('DEBUG: debug parsers:')
        print('DEBUG: args.testrun:',args.testrun)
        print('DEBUG: args.nseed:',args.nseed)
        print('DEBUG: args.log_int:',args.log_int,'\n')

        print('DEBUG: parallel parsers:')
        print('DEBUG: args.backend:',args.backend)
        print('DEBUG: args.nworker:',args.nworker)
        print('DEBUG: args.prefetch:',args.prefetch)
        print('DEBUG: args.cuda:',args.cuda)
        print('DEBUG: args.benchrun:',args.benchrun,'\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

# load datasets
    turb_data1 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1000',\
        loader=hdf5_loader, extensions='.hdf5')
    turb_data2 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1200',\
        loader=hdf5_loader, extensions='.hdf5')
    turb_data3 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1600',\
        loader=hdf5_loader, extensions='.hdf5')
    turb_data4 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width3000',\
         loader=hdf5_loader, extensions='.hdf5')

    turb_data = torch.utils.data.ConcatDataset([turb_data1, turb_data2, turb_data3, turb_data4])

    #turb_data = datasets.DatasetFolder(args.data_dir+'trainfolder/Width10_12_16_18',\
    #     loader=hdf5_loader, extensions='.hdf5')

    test_data = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1800',\
         loader=hdf5_loader, extensions='.hdf5')


    #load/specify mesh spacing data
    del_x = 0.2598
    #del_y = np.loadtxt(args.data_dir+'deltay.dat')
    coords_y = np.loadtxt(args.data_dir+'coords_y.dat')
    del_z = 0.0866

    # restricts data loading to a subset of the dataset exclusive to the current process
    args.shuff = args.shuff and not args.testrun
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        turb_data, num_replicas=gwsize, rank=lrank, shuffle = args.shuff)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data, num_replicas=gwsize, rank=lrank, shuffle = args.shuff)

# distribute dataset to workers
    # persistent workers is not possible for nworker=0
    pers_w = False if args.nworker>1 else False

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    train_loader = torch.utils.data.DataLoader(turb_data, batch_size=args.batch_size,
        sampler=train_sampler, collate_fn=collate_batch, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
        sampler=test_sampler, collate_fn=collate_batch, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n') 

    # create model
    model = autoencoder().to(device)

# distribute model to workers
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model,\
            device_ids=[device], output_device=device)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model)

# optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    #scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

# resume state 
    start_epoch = 0
    best_acc = np.Inf
    res_name='checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
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

    if start_epoch>=args.epochs:
        if grank==0:
            print(f'WARNING: given epochs are less than the one in the restart file!\n'
                  f'WARNING: SYS.EXIT is issued')
        dist.destroy_process_group()
        sys.exit()

# start trainin loop
    et = time.time()
    for epoch in range(start_epoch, args.epochs):
        lt = time.time()
        # training
        if args.benchrun and epoch==args.epochs-1:
            # profiling (done on last epoch - slower!)
            with torch.autograd.profiler.profile(use_cuda=args.cuda, profile_memory=True) as prof:
                loss_acc = train(distrib_model, train_sampler, loss_function, \
                            device, train_loader, optimizer, epoch, grank, lt, scheduler_lr, del_x, coords_y, del_z)
        else:
            loss_acc = train(distrib_model, train_sampler, loss_function, \
                            device, train_loader, optimizer, epoch, grank, lt, scheduler_lr, del_x, coords_y, del_z)

        # save first epoch timer
        if epoch == start_epoch:
            first_ep_t = time.time()-lt

        # printout profiling results of the last epoch
        if args.benchrun and epoch==args.epochs-1 and grank==0:
            print(f'\n--------------------------------------------------------') 
            print(f'DEBUG: benchmark of last epoch:\n')
            what1 = 'cuda' if args.cuda else 'cpu'
            print(prof.key_averages().table(sort_by='self_'+str(what1)+'_time_total'))

        # save state if found a better state 
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0 and not args.benchrun:
            save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,is_best)
            # reset best_acc
            best_acc = min(loss_acc, best_acc)

# finalise training
    # save final state
    if not args.benchrun:
        save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,True)
    dist.barrier()

    # some debug
    if grank==0:
        print(f'\n--------------------------------------------------------') 
        print(f'DEBUG: training results:')
        print(f'TIMER: first epoch time: {first_ep_t} s')
        print(f'TIMER: last epoch time: {time.time()-lt} s')
        print(f'TIMER: total epoch time: {time.time()-et} s')
        print(f'TIMER: average epoch time: {(time.time()-et)/args.epochs} s')
        if epoch > 1:
            print(f'TIMER: total epoch-1 time: {time.time()-et-first_ep_t} s')
            print(f'TIMER: average epoch-1 time: {(time.time()-et-first_ep_t)/(args.epochs-1)} s')
        if args.benchrun:
            print(f'TIMER: total epoch-2 time: {lt-first_ep_t} s')
            print(f'TIMER: average epoch-2 time: {(lt-first_ep_t)/(args.epochs-2)} s')
        print('DEBUG: memory req:',int(torch.cuda.memory_reserved(lrank)/1024/1024),'MB') \
                if args.cuda else 'DEBUG: memory req: - MB'
        print('DEBUG: memory summary:\n\n',torch.cuda.memory_summary(0)) if args.cuda else ''

# start testing loop
    avg_mean_sqr_diff = test(distrib_model, loss_function, device, test_loader, grank, gwsize, del_x, coords_y, del_z)

# clean-up
    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s')
    dist.destroy_process_group()

if __name__ == "__main__": 
    main()
    sys.exit()

# eof
