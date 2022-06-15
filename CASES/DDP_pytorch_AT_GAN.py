#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: RS
# version: 211029a

# std libs
import argparse, sys, platform, os, time, numpy as np, h5py, random, shutil, re
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
    plt.savefig('vfield_recon_GAN'+data_org+str(random.randint(0,100))+'.pdf',
                    bbox_inches = 'tight', pad_inches = 0)


#def collate_batch(batch):
#    imgs = [item[0].reshape(int(item[0].shape[0]/89), 89, item[0].shape[1], \
#        item[0].shape[2], item[0].shape[3]) for item in batch]
#    targets = [item[1] for item in batch]
#    imgs = torch.cat((imgs))
#    return imgs, targets

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
        data_out = data
    return data_out

# loader for turbulence HDF5 data
def hdf5_loader(path, max_time=636976, time_history=5, time_step=24):
    time_ins = int(re.findall(r'\d+',path.split('/')[-1].split('.')[0])[0])
    if (max_time-time_ins)<=time_step*(time_history-1):
        time_ins = time_ins-(time_step*(time_history)-(max_time-time_ins))
    time_ins = list(range(time_ins,time_ins+time_step*(time_history+1),time_step))
    data_f = []
    for i in range(len(time_ins)):
        path_f = '/'.join(path.split('/')[:-1])+'/boxOutput'+str(time_ins[i])+'.hdf5'
        f = h5py.File(path_f, 'r')
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
        data_u = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['u'])).permute((1,0,2))[:-59,:,:]
        data_u = 2*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1
        data_v = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['v'])).permute((1,0,2))[:-59,:,:]
        data_v = 2*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1
        data_w = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['w'])).permute((1,0,2))[:-59,:,:]
        data_w = 2*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1
        data_f.append(torch.reshape(torch.cat((data_u, data_v, data_w)),
               (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3)))
    return torch.cat(data_f[:-1],1), data_f[-1]

def hdf5_loader_test(path, max_time=640000, time_history=5, time_step=24):
    time_ins = int(re.findall(r'\d+',path.split('/')[-1].split('.')[0])[0])
    if (max_time-time_ins)<=time_step*(time_history-1):
        time_ins = time_ins-(time_step*(time_history)-(max_time-time_ins))
    time_ins = list(range(time_ins,time_ins+time_step*(time_history+1),time_step))
    data_f = []
    for i in range(len(time_ins)):
        path_f = '/'.join(path.split('/')[:-1])+'/boxOutput'+str(time_ins[i])+'.hdf5'
        f = h5py.File(path_f, 'r')
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
        data_u = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['u'])).permute((1,0,2))[:-59,:,:]
        data_u = 2*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1
        data_v = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['v'])).permute((1,0,2))[:-59,:,:]
        data_v = 2*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1
        data_w = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['w'])).permute((1,0,2))[:-59,:,:]
        data_w = 2*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1
        data_f.append(torch.reshape(torch.cat((data_u, data_v, data_w)),
               (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3)))
    return torch.cat(data_f[:-1],1), data_f[-1]

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

# Generator 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3*5, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(8)
        self.conv5 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inp_d):
        conv1 = self.bn1(self.conv1(inp_d))
        conv1 = interpolate(conv1, scale_factor=2, mode='bilinear', align_corners=True)
        conv1 = pad(conv1,(0,0,0,1))
        conv1 = self.leaky_reLU(conv1)

        conv2 = self.bn2(self.conv2(conv1))
        conv2 = interpolate(conv2, scale_factor=2, mode='bilinear', align_corners=True)
        conv2 = self.leaky_reLU(conv2)

        conv3 = self.bn3(self.conv3(conv2))
        conv3 = interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        conv3 = pad(conv3,(0,1,0,1)) 
        conv3 = self.leaky_reLU(conv3)

        conv3 = self.bn3(self.conv3(conv2))
        conv3 = interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        conv3 = pad(conv3,(0,1,0,1)) 
        conv3 = self.leaky_reLU(conv3)

        conv4 = self.bn4(self.conv4(conv3))
        conv4 = interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)
        conv4 = self.leaky_reLU(conv4)

        return self.conv5(conv4)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

        # input is (nc) x 250 x 402
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=3, padding=[1,0], bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=3, padding=[2,0], bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, inp_x):
        conv1 = self.leaky_reLU(self.bn1(self.conv1(inp_x)))
        conv2 = self.leaky_reLU(self.bn2(self.conv2(conv1)))
        conv3 = self.leaky_reLU(self.bn3(self.conv3(conv2)))
        conv4 = self.leaky_reLU(self.bn4(self.conv4(conv3)))

        return self.sigm(self.conv5(conv4))

# weights initialization from a Normal distribution with mean=0, stdev=0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# save state of the training
def save_state(epoch,distrib_model_gen,distrib_model_dis,gloss_acc,dloss_acc,optimizer_gen,optimizer_dis,res_name,grank,gwsize,is_best):
    rt = time.time()
    # find if is_best happened in any worker
    is_best_m = par_allgather_obj(is_best,gwsize)

    if any(is_best_m):
        # find which rank is_best happened - select first rank if multiple
        is_best_rank = np.where(np.array(is_best_m)==True)[0][0]

        # collect state
        state = {'epoch': epoch + 1,
                 'state_dict_gen': distrib_model_gen.state_dict(),
                 'state_dict_dis': distrib_model_dis.state_dict(),
                 'best_acc_g': gloss_acc,
                 'best_acc_d': dloss_acc,
                 'optimizer_gen' : optimizer_gen.state_dict(),
                 'optimizer_dis' : optimizer_dis.state_dict()}

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
def train(model_gen, model_dis, sampler, criterion, device, train_loader, optimizer_gen, optimizer_dis, epoch, grank, lt, schedulerlr_gen, schedulerlr_dis):
    count=0

    # Lists to keep track of progress
    img_list = []
    gen_losses = []
    dis_losses = []
    iters = 0
    real_label = 1.
    fake_label = 0.

    genl_acc = 0.0
    disl_acc = 0.0

    #sampler.set_epoch(epoch)
    for inputs, labels in train_loader:
        inps = inputs[0].view(1, -1, *(inputs[0].size()[2:])).squeeze(0).float().to(device)
        itargs = inputs[1].view(1, -1, *(inputs[1].size()[2:])).squeeze(0).float().to(device)
        inps = interpolate(inps, scale_factor=0.0625, mode='bilinear', align_corners=True)
        # ===================forward=====================
        ## Train with all-real batch
        optimizer_dis.zero_grad()
        b_size = itargs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        predictions = model_dis(itargs).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(predictions, label)
        # ===================backward====================
        # Calculate gradients for D in backward pass
        errD_real.backward()                                       # Backward pass
        D_x = predictions.mean().item()
        #optimizer.step()                                       # Optimizer step
        ## Train with all-fake batch
        # Batch of latent vectors from earlier time steps
        # Generate fake image batch with G
        fake = model_gen(inps)
        label.fill_(fake_label)
        # Classify all fake batch with D
        predictions = model_dis(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(predictions, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = predictions.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizer_dis.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer_gen.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # another forward pass of all-fake batch through D
        predictions = model_dis(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(predictions, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = predictions.mean().item()
        # Update G
        optimizer_gen.step()

        genl_acc+= errG.item()
        disl_acc+= errD.item()
        # Output training stats
        if count % args.log_int == 0 and grank==0:
            print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):3.2f}% complete'\
                  f' / {time.time() - lt:.2f} s, Loss_D: {errD.item():2.4f}, \
                  Loss_G: {errG.item():2.4f}, D(x): {D_x:2.4f}, D(G(z)): {D_G_z1:2.4f} \
                  / {D_G_z2:2.4f}')
                
        # Check how the generator is doing by saving G's output on fixed_noise
        #if (count % 500 == 0) or ((epoch == args.epochs-1) and (count == len(train_loader)-1)):
        #    with torch.no_grad():
        #        fake = model_gen(fixed_noise).detach().cpu()
        #    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        count+=1

        # Save Losses for plotting later
        gen_losses.append(errG.item())
        dis_losses.append(errD.item())


    if args.schedule:
        schedulerlr_gen.step()
        schedulerlr_dis.step()

    # profiling statistics
    if grank==0:
        print('TIMER: epoch time:', time.time()-lt, 's\n')

    return gen_losses, dis_losses, genl_acc, disl_acc

# test loop
def test(model_gen, loss_function, device, test_loader, grank, gwsize):
    et = time.time()
    model_gen.eval()
    test_loss = 0.0
    mean_sqr_diff = []
    count=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inps = inputs[0].view(1, -1, *(inputs[0].size()[2:])).squeeze(0).float().to(device)
            targs = inputs[1].view(1, -1, *(inputs[1].size()[2:])).squeeze(0).float().to(device)
            inps = interpolate(inps, scale_factor=0.0625, mode='bilinear', align_corners=True)        
            
            predictions = model_gen(inps)
            loss = loss_function(predictions.float(), targs.float())
            test_loss+= loss.item()/targs.shape[0]
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            mean_sqr_diff.append(\
                torch.mean(torch.square(predictions.float()-targs.float())).item())
            count+=1

    # mean from dataset (ignore if just 1 dataset)
    if count>1:
        mean_sqr_diff=np.mean(mean_sqr_diff)

    if grank==0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time()-et} s')
        if not args.testrun or not args.benchrun:
            plot_scatter(targs[0][0].cpu().detach().numpy(), 
                    predictions[0][0].cpu().detach().numpy(), 'test')

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
    turb_data = datasets.DatasetFolder(args.data_dir+'trainfolder',\
        loader=hdf5_loader, extensions='.hdf5')
    test_data = datasets.DatasetFolder(args.data_dir+'testfolder',\
         loader=hdf5_loader_test, extensions='.hdf5')

    # restricts data loading to a subset of the dataset exclusive to the current process
    args.shuff = args.shuff and not args.testrun
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        turb_data, num_replicas=gwsize, rank=grank, shuffle = args.shuff)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data, num_replicas=gwsize, rank=grank, shuffle = args.shuff)

# distribute dataset to workers
    # persistent workers is not possible for nworker=0
    pers_w = True if args.nworker>1 else False

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    train_loader = torch.utils.data.DataLoader(turb_data, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n') 

    # create model
    model_gen = Generator().to(device)
    model_dis = Discriminator().to(device)

# distribute model to workers
    if args.cuda:
        distrib_model_gen = torch.nn.parallel.DistributedDataParallel(model_gen,\
            device_ids=[device], output_device=device)
        distrib_model_dis = torch.nn.parallel.DistributedDataParallel(model_dis,\
            device_ids=[device], output_device=device)
    else:
        distrib_model_gen = torch.nn.parallel.DistributedDataParallel(model_gen)
        distrib_model_dis = torch.nn.parallel.DistributedDataParallel(model_dis)

    distrib_model_gen.apply(weights_init)
    distrib_model_dis.apply(weights_init)

# optimizer
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    test_loss_function = nn.MSELoss()
    # Batch of latent vectors to visualize generator progression
    fixed_noise = torch.randn(args.batch_size, 3*5, 15, 25, device=device)

    #loss_function = nn.MSELoss()
    optimizer_gen = torch.optim.Adam(distrib_model_gen.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer_dis = torch.optim.Adam(distrib_model_dis.parameters(), lr=args.lr, weight_decay=args.wdecay)

    schedulerlr_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=args.gamma)
    schedulerlr_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer_dis, gamma=args.gamma)
    #scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

# resume state 
    start_epoch = 0
    best_acc_g = np.Inf
    best_acc_d = np.Inf
    res_name='checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(program_dir+'/'+res_name, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc_g = checkpoint['best_acc_g']
            best_acc_d = checkpoint['best_acc_d']
            distrib_model_gen.load_state_dict(checkpoint['state_dict_gen'])
            optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
            distrib_model_dis.load_state_dict(checkpoint['state_dict_dis'])
            optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
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
    # Lists to keep track of progress
    
    et = time.time()
    for epoch in range(start_epoch, args.epochs):
        lt = time.time()
        # training

        if args.benchrun and epoch==args.epochs-1:
            # profiling (done on last epoch - slower!)
            with torch.autograd.profiler.profile(use_cuda=args.cuda, profile_memory=True) as prof:
                gen_l, dis_l, gloss_acc, dloss_acc = train(distrib_model_gen, distrib_model_dis, train_sampler, criterion, \
                            device, train_loader, optimizer_gen, optimizer_dis, epoch, grank, lt, schedulerlr_gen, schedulerlr_dis)
        else:
            gen_l, dis_l, gloss_acc, dloss_acc = train(distrib_model_gen, distrib_model_dis, train_sampler, criterion, \
                            device, train_loader, optimizer_gen, optimizer_dis, epoch, grank, lt, schedulerlr_gen, schedulerlr_dis)

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
        is_best = gloss_acc < best_acc_g or dloss_acc < best_acc_d
        if epoch % args.restart_int == 0 and not args.benchrun:
            save_state(epoch,distrib_model_gen,distrib_model_dis,gloss_acc,dloss_acc,optimizer_gen,optimizer_dis,res_name,grank,gwsize,is_best)
            # reset best_acc
            best_acc_g = min(gloss_acc, best_acc_g)
            best_acc_d = min(dloss_acc, best_acc_d)

# finalise training
    # save final state
    if not args.benchrun:
        save_state(epoch,distrib_model_gen,distrib_model_dis,gloss_acc,dloss_acc,optimizer_gen,optimizer_dis,res_name,grank,gwsize,True)
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
    avg_mean_sqr_diff = test(distrib_model_gen, test_loss_function, device, test_loader, grank, gwsize)

# clean-up
    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s')
    dist.destroy_process_group()

if __name__ == "__main__": 
    main()
    sys.exit()

# eof
