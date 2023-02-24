#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to train custom diffusion model with large actuated TBL dataset.
instead of noise, dataset is filtered with Gaussian filter and artificial turbulence is added.
authors: EI, RS
version: 230224a
notes: for cost Perlin noise is selected, use spectral methods later on.
works for non-actuated case at this moment!
"""

# std libs
import argparse, sys, platform, os, time, h5py, random, shutil, logging
import matplotlib.pyplot as plt, numpy as np, scipy as sp
from perlin_noise import PerlinNoise

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import datasets, transforms

# parsed settings
def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='PyTorch-DDP actuated TBL')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the'
                        ' local filesystem (default: ./)')
    parser.add_argument('--restart-int', type=int, default=100,
                        help='restart interval per epoch (default: 100)')
    parser.add_argument('--concM', type=int, default=1,
                        help='increase dataset size with this factor (default: 1)')

    # model parsers
    parser.add_argument('--batch-size', type=int, default=1, choices=range(1,int(1e7)), metavar="[1-1e9]",
                        help='input batch size for training (default: 1, min: 1, max: 1e9)')
    parser.add_argument('--epochs', type=int, default=10, choices=range(1,int(1e7)), metavar="[1-1e9]",
                        help='number of epochs to train (default: 10, min: 1, max: 1e9)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
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
    parser.add_argument('--skipplot', action='store_true', default=False,
                        help='skips test postprocessing (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training (default: 10)')

    # parallel parsers
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--nworker', type=int, default=0,
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs (default: False)')

    # benchmarking parsers
    parser.add_argument('--synt', action='store_true', default=False,
                        help='use a synthetic dataset instead (default: False)')
    parser.add_argument('--synt-dpw', type=int, default=1000, choices=range(1,int(1e7)), metavar="[1-1e9]",
                        help='dataset size per GPU if synt is true (default: 1000, min: 1, max: 1e9)')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    # optimizations
    parser.add_argument('--cudnn', action='store_true', default=False,
                        help='turn on cuDNN optimizations (default: False)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='turn on Automatic Mixed Precision (default: False)')
    parser.add_argument('--reduce-prec', action='store_true', default=False,
                        help='reduce precision of the dataset for faster I/O (default: False)')
    parser.add_argument('--scale-lr', action='store_true', default=False,
                        help='scale lr with #workers (default: False)')
    parser.add_argument('--accum-iter', type=int, default=1,
                        help='accumulate gradient update (default: 1 - turns off)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs

# postproc
def plot_scatter(org_img, inp_img, out_img, epoch, sigma, final=False):
    fig = plt.figure(figsize = (4,12))
    plt.rc('font', size=10)
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(org_img,interpolation='None')
    ax1.set_title('Original')
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(inp_img, interpolation='None')
    ax2.set_title('Input')
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(out_img, interpolation='None')
    ax3.set_title('Output')

    outName = 'vfield_recon_DM_train_'+str(epoch)+'_'+str(sigma)
    plt.savefig(outName+'.pdf',bbox_inches='tight',pad_inches=0)

    # export the data at final
    if final:
        h5f = h5py.File(outName+'.hdf5', 'w')
        h5f.create_dataset('orig', data=org_img)
        h5f.create_dataset('input', data=inp_img)
        h5f.create_dataset('output', data=out_img)
        h5f.close()

# loader for turbulence HDF5 data
class custom_batch:
    def __init__(self, batch):
        inp = [item[0].reshape(int(item[0].shape[0]/89), 89, item[0].shape[1], \
            item[0].shape[2], item[0].shape[3]) for item in batch]
        self.inp = torch.cat((inp))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        return self

def collate_batch(batch):
    return custom_batch(batch)

def uniform_data(data):
    if data.shape[2]==300 or data.shape[2]==400 or data.shape[2]==450:
        data_out = data.view((data.shape[0],data.shape[1],data.shape[2]//50,50,data.shape[3]))
        data_out = torch.cat((data_out[:,:,:5,:,:].view((data.shape[0],data.shape[1],250,data.shape[3])), \
                             data_out[:,:,-5:,:,:].view((data.shape[0],data.shape[1],250,data.shape[3]))))
    elif data.shape[2]==750:
        data_out = data.view((data.shape[0],data.shape[1],data.shape[2]//50,50,data.shape[3]))
        data_out = torch.cat((data_out[:,:,:5,:,:].view((data.shape[0],data.shape[1],250,data.shape[3])), \
                            data_out[:,:,5:10,:,:].view((data.shape[0],data.shape[1],250,data.shape[3])), \
                             data_out[:,:,10:,:,:].view((data.shape[0],data.shape[1],250,data.shape[3]))))
    else:
        data_out = data
    return data_out

def hdf5_loader(path):
    # read hdf5 w/ structures
    f = h5py.File(path, 'r')
    f1 = f[list(f.keys())[0]]
    data_u = np.array(f1[list(f1.keys())[0]]['u'])
    data_v = np.array(f1[list(f1.keys())[0]]['v'])
    data_w = np.array(f1[list(f1.keys())[0]]['w'])

    # convert to torch and remove last ten layers assuming free stream
    dtype = torch.float16 if args.reduce_prec else torch.float32
    data_u = torch.tensor(data_u,dtype=dtype).permute((1,0,2))[:-9,:,:]
    data_v = torch.tensor(data_v,dtype=dtype).permute((1,0,2))[:-9,:,:]
    data_w = torch.tensor(data_w,dtype=dtype).permute((1,0,2))[:-9,:,:]

    # normalise data to -1:1
    data_u = 2.0*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1.0
    data_v = 2.0*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1.0
    data_w = 2.0*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1.0

    return uniform_data(torch.cat((data_u, data_v, data_w)).view(( \
            3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3)))

# synthetic data for benchmarking
class SyntheticDataset_train(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        data = torch.randn(89, 3, 192, 248)
        target = random.randint(0, 999)
        return (data, target)

    def __len__(self):
        return args.batch_size * gwsize * args.synt_dpw

class SyntheticDataset_test(SyntheticDataset_train):
    def __len__(self):
        return args.batch_size * gwsize

# debug of the run
def debug_ini(timer):
    if grank==0:
        logging.basicConfig(format='%(levelname)s: %(message)s', stream=sys.stdout, level=logging.INFO)
        logging.info('initialise: '+str(timer)+'s')
        logging.info('local ranks: '+str(lwsize)+'/ global ranks:'+str(gwsize))
        logging.info('sys.version: '+str(sys.version))
        logging.info('parsers list:')
        list_args = [x for x in vars(args)]
        for count,name_args in enumerate(list_args):
            logging.info('args.'+name_args+': '+str(vars(args)[list_args[count]]))

        # add warning here!
        warning1=False
        if args.benchrun and args.epochs<3:
            logging.warning('benchrun requires atleast 3 epochs - setting epochs to 3\n')
            warning1=True
        if not warning1:
            logging.warning('all OK!\n')

    return logging

# debug of the training
def debug_final(logging,start_epoch,last_epoch,first_ep_t,last_ep_t,tot_ep_t):
    if grank==0:
        done_epochs = last_epoch - start_epoch + 1
        print(f'\n--------------------------------------------------------')
        logging.info('training results:')
        logging.info('first epoch time: {:.2f}'.format(first_ep_t)+' s')
        logging.info('last epoch time: {:.2f}'.format(last_ep_t)+' s')
        logging.info('total epoch time: {:.2f}'.format(tot_ep_t)+' s')
        logging.info('average epoch time: {:.2f}'.format(tot_ep_t/done_epochs)+' s')
        if done_epochs>1:
            tot_ep_tm1 = tot_ep_t - first_ep_t
            logging.info('total epoch-1 time: {:.2f}'.format(tot_ep_tm1)+' s')
            logging.info('average epoch-1 time: {:.2f}'.format(tot_ep_tm1/(done_epochs-1))+' s')
        if args.benchrun and done_epochs>2:
            tot_ep_tm2 = tot_ep_t - first_ep_t - last_ep_t
            logging.info('total epoch-2 time: {:.2f}'.format(tot_ep_tm2)+' s')
            logging.info('average epoch-2 time: {:.2f}'.format(tot_ep_tm2/(done_epochs-2))+' s')
        # memory on worker 0
        if args.cuda:
            logging.info('memory req: '+str(int(torch.cuda.max_memory_reserved(0)/1024/1024))+' MB')
            logging.info('memory summary:\n'+str(torch.cuda.memory_summary(0)))

# network
"""
U-NET from
https://github.com/milesial/Pytorch-UNet/blob/master/unet/
"""
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# save state of the training
def save_state(epoch,distrib_model,loss_acc,optimizer,res_name,is_best):
    rt = time.perf_counter()
    # find if is_best happened in any worker
    is_best_m = par_allgather_obj(is_best)

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
            #shutil.copyfile(res_name, res_name+'_'+str(epoch))
            logging.info('state in '+str(grank)+' is saved on epoch:'+str(epoch)+\
                    ' in '+str(time.perf_counter()-rt)+' s')

# deterministic dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def trace_handler(prof):
    # do operations when a profiler calles a trace
    #prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")
    if grank==0:
        logging.info('profiler called a trace')

# diffusion model for ATB
class diff_model:
    def __init__(self,inputs,noise_map,epoch=1,sigma=None,eps=1e-2):
        # start a timer
        lt = time.perf_counter()

        # stacks,u_i,n_x,n_y
        m,n,a,b = inputs.shape[:]

        # level of diffusion over epochs (modify if sigma is not given in arg)
        if sigma is None:
            sigma = 1.0 if epoch >     0 else sigma
            sigma = 2.0 if epoch >  4000 else sigma
            sigma = 3.0 if epoch >  8000 else sigma
            sigma = 4.0 if epoch > 12000 else sigma
            sigma = 5.0 if epoch > 16000 else sigma
        self.sigma = sigma

        # generate Perlin noise only if dimensions of input changed (expensive)
        try:
            try_e = inputs[0,0,:,:] + noise_map
        except ValueError:
            noise_map = noise_gen(a,b)
            if grank==0:
                logging.info('noise regenerated!')

        # apply filter with added noise for diffusion
        self.inputs_dm = torch.clone(inputs)
        if sigma>1.01:
            # 2D-Gaussian filter with sigma (size=2*r+1, w/ r=round(sigma,truncate), truncate=1 to get desired r)
            for i in range(m):
                for j in range(n):
                    res = sp.ndimage.gaussian_filter(inputs[i,j,:,:], self.sigma, truncate=1.0)
                    self.inputs_dm[i,j,:,:] = torch.from_numpy(res + noise_map*eps)

        self.et = time.perf_counter()-lt

# noise generator
def noise_gen(xpix,ypix):
    # Perlin noise
    noise = PerlinNoise(octaves=10)
    noise_map = [[noise([i/xpix, j/ypix]) for i in range(ypix)] for j in range(xpix)]
    return np.array(noise_map)

# train loop
def train(model, sampler, loss_function, device, train_loader, optimizer, epoch, scheduler, noise_map):
    # start timers
    lt_1 = time.perf_counter()

    # profiler
    """
    - activities (iterable): list of activity groups (CPU, CUDA) to use in profiling,
    supported values: torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA.
    Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.
    - schedule (callable): callable that takes step (int) as a single parameter and returns
    ProfilerAction value that specifies the profiler action to perform at each step.
        the profiler will skip the first ``skip_first`` steps,
        then wait for ``wait`` steps,
        then do the warmup for the next ``warmup`` steps,
        then do the active recording for the next ``active`` steps and
        then repeat the cycle starting with ``wait`` steps.
        The optional number of cycles is specified with the ``repeat`` parameter,
           0 means that the cycles will continue until the profiling is finished.
    - on_trace_ready (callable): callable that is called at each step
    when schedule returns ProfilerAction.RECORD_AND_SAVE during the profiling.
    - record_shapes (bool): save information about operator's input shapes.
    - profile_memory (bool): track tensor memory allocation/deallocation.
    - with_stack (bool): record source information (file and line number) for the ops.
    - with_flops (bool): use formula to estimate the FLOPs (floating point operations)
    of specific operators (matrix multiplication and 2D convolution).
    - with_modules (bool): record module hierarchy (including function names) corresponding
    to the callstack of the op. e.g. If module A's forward call's module B's forward
    which contains an aten::add op, then aten::add's module hierarchy is A.B
    Note that this support exist, at the moment, only for TorchScript models and not eager mode models.
    """
    if args.benchrun:
        # profiler options
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # at least 3 epochs required with
            # default wait=1, warmup=1, active=args.epochs, repeat=1, skip_first=0
            schedule=torch.profiler.schedule(wait=1,warmup=1,active=1,repeat=1,skip_first=0),
            #on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            on_trace_ready=trace_handler,
            record_shapes=False,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=False
        )
        # profiler start
        prof.start()

    loss_acc = 0.0
    lt_2 = 0.0
    sampler.set_epoch(epoch)
    for batch_ndx, (samples) in enumerate(train_loader):
        do_backprop = ((batch_ndx + 1) % args.accum_iter == 0) or (batch_ndx + 1 == len(train_loader))
        inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float()

        # diffusion model
        lt_3 = time.perf_counter()
        data = diff_model(inputs, noise_map, epoch)
        lt_2 += time.perf_counter() - lt_3

        # adjust LR with sigma just testing!
        adjust_learning_rate(optimizer, epoch, data.sigma)

        # train part
        with torch.set_grad_enabled(True):
            if args.amp:
                with torch.cuda.amp.autocast():
                    # forward pass
                    predictions = model(data.inputs_dm.to(device)).float()
                    loss = loss_function(predictions, inputs.to(device)) / args.accum_iter
                    # backward pass
                    loss.backward()
                    if do_backprop:
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                predictions = model(data.inputs_dm.to(device)).float()
                loss = loss_function(predictions, inputs.to(device)) / args.accum_iter
                loss.backward()
                if do_backprop:
                    optimizer.step()
                    optimizer.zero_grad()

        loss_acc+= loss.item()

        if batch_ndx % args.log_int == 0 and grank==0:
            print(f'Epoch: {epoch} / {100 * (batch_ndx + 1) / len(train_loader):06.2f}% done',\
                  f'/ loss: {loss_acc:19.16f} / sigma={data.sigma}', end='')
            print(f' / bp: {do_backprop}') if not do_backprop else print(f'')

        # profiler step per batch
        if args.benchrun:
            prof.step()

    # TEST w/ plots
    if grank==0 and epoch%100==0:
        plot_scatter(inputs[0][0].cpu().detach().numpy(), \
                data.inputs_dm[0][0].cpu().detach().numpy(), \
                predictions[0][0].cpu().detach().numpy(), epoch, data.sigma)

    # lr scheduler
    if args.schedule:
        scheduler.step()

    # profiler end
    if args.benchrun:
        prof.stop()

    # timer for current epoch
    if grank==0:
        logging.info('accumulated lost: {:19.16f}'.format(loss_acc))
        logging.info('epoch time: {:.2f}'.format(time.perf_counter()-lt_1)+' s')
        logging.info('filter time: {:.2f}'.format(lt_2)+' s ({:3.2f}'.\
                format(100*lt_2/(time.perf_counter()-lt_1))+'% of epoch time)\n')

    # printout profiler
    if args.benchrun and epoch==args.epochs-1 and grank==0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: benchmark of last epoch:\n')
        what1 = 'cuda' if args.cuda else 'cpu'
        print(prof.key_averages().table(sort_by='self_'+str(what1)+'_time_total', row_limit=-1))

    return loss_acc, time.perf_counter()-lt_1

# test loop
def test(model, loss_function, device, test_loader, noise_map):
    et = time.perf_counter()

    # testing various sigmas
    for sigma_test in [1,2,3,4,5,10,20]:

        model.eval()
        test_loss = 0.0
        mean_sqr_diff = 0.0
        with torch.no_grad():
            for count, (samples) in enumerate(test_loader):
                inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float().to(device)

                # test highly diffused image for diffusion model
                data = diff_model(inputs.cpu(), noise_map, sigma=sigma_test)
                predictions = model(data.inputs_dm.to(device)).float()
                loss = loss_function(predictions, inputs.to(device)) / args.accum_iter
                test_loss+= torch.nan_to_num(loss).item()/inputs.shape[0]
                # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
                res = torch.mean(torch.square(torch.nan_to_num(predictions)-torch.nan_to_num(inputs)))
                res[torch.isinf(res)] = 0.0
                mean_sqr_diff = mean_sqr_diff*count/(count+1.0) + res.item()/(count+1.0)

        # mean from gpus
        avg_test_loss = float(par_mean(test_loss).cpu())
        avg_mean_sqr_diff = float(par_mean(mean_sqr_diff).cpu())
        if grank==0:
            print(f'testing results for sigma {sigma_test}:')
            print(f'avg_test_loss: {avg_test_loss}')
            print(f'avg_mean_sqr_diff: {avg_mean_sqr_diff} m**2/s**2\n')

        # post-process
        if grank==0 and not args.skipplot and not args.testrun and not args.benchrun:
            plot_scatter(inputs[0][0].cpu().detach().numpy(), \
                    data.inputs_dm[0][0].cpu().detach().numpy(), \
                    predictions[0][0].cpu().detach().numpy(), \
                    epoch=args.epochs, sigma=sigma_test, final=True)

    if grank==0:
        logging.info('total testing time: {:.2f}'.format(time.perf_counter()-et)+' s')

# encode export
def encode_exp(encode, device, train_loader):
    for batch_ndx, (samples) in enumerate(train_loader):
        inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float().to(device)
        predictions = encode(inputs).float()

        # export the data
        inp = inputs.to('cpu').detach().numpy()
        h5f = h5py.File('./test_'+str(batch_ndx)+'_'+str(grank)+'.h5', 'w')
        h5f.create_dataset('input', data=inputs.cpu().detach().numpy())
        h5f.create_dataset('prediction', data=predictions.cpu().detach().numpy())
        h5f.close()
        break

def adjust_learning_rate(optimizer, sigma, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr*factor

# PARALLEL HELPERS
# sum of field over GPGPUs
def par_sum(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    return res

# mean of field over GPGPUs
def par_mean(field):
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
def par_allgather(field):
    if args.cuda:
        sen = torch.Tensor([field]).cuda()
        res = [torch.Tensor([field]).cuda() for i in range(gwsize)]
    else:
        sen = torch.Tensor([field])
        res = [torch.Tensor([field]) for i in range(gwsize)]
    dist.all_gather(res,sen,group=None)
    return res

# gathers any object from the whole group in a list (to all workers)
def par_allgather_obj(obj):
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

    # start the time for profiling
    st = time.perf_counter()

# initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend=args.backend)

# deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

# get job rank info -- rank==0 master gpu
    global lwsize, gwsize, grank, lrank
    lwsize = torch.cuda.device_count() if args.cuda else 0 # local world size - per node (0 if CPU run)
    gwsize = dist.get_world_size()     # global world size - per run
    grank = dist.get_rank()            # global rank - assign per run
    lrank = dist.get_rank()%lwsize     # local rank - assign per node

    # debug of the run
    logging = debug_ini(time.perf_counter()-st)

    # set workers and apply testrun status
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

    # cuDNN optimisation
    torch.backends.cudnn.benchmark = args.cudnn

# load datasets
    if args.synt:
        # synthetic dataset if selected
        turb_data = SyntheticDataset_train()
        test_data = SyntheticDataset_test()
    else:
        turb_data = datasets.DatasetFolder(args.data_dir+'trainfolder',\
            loader=hdf5_loader, extensions='.hdf5')
        test_data = datasets.DatasetFolder(args.data_dir+'testfolder',\
             loader=hdf5_loader, extensions='.hdf5')

# increase dataset size if required
    largeData = []
    for i in range(args.concM):
        largeData.append(turb_data)
    turb_data = torch.utils.data.ConcatDataset(largeData)

    # concat data
    train_dataset = torch.utils.data.ConcatDataset(largeData)

    # restricts data loading to a subset of the dataset exclusive to the current process
    args.shuff = args.shuff and not args.testrun
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        turb_data, num_replicas=gwsize, rank=grank, shuffle = args.shuff)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data, num_replicas=gwsize, rank=grank, shuffle = args.shuff)

# distribute dataset to workers
    # persistent workers is not possible for nworker=0
    #pers_w = True if args.nworker>1 else False
    pers_w = False # GPU Xid 119 error?

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    train_loader = torch.utils.data.DataLoader(turb_data, batch_size=args.batch_size,
        sampler=train_sampler, collate_fn=collate_batch, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
        sampler=test_sampler, collate_fn=collate_batch, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )

    if grank==0:
        logging.info('read data: '+str(time.perf_counter()-st)+'s\n')

    # create model
    model = UNet().to(device)

# distribute model to workers
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model,\
            device_ids=[device], output_device=device)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model)

    # scale lr with #workers
    lr_scale = gwsize if args.scale_lr else 1

# optimizer
    loss_function = nn.MSELoss()
    #optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr*lr_scale, weight_decay=args.wdecay)
    optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr*lr_scale)

# scheduler
    #scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000)

    # used lr and info on num. of parameters
    if grank==0:
        logging.info('current learning rate: '+str(args.lr*lr_scale)+'\n')
        tp_d = sum(p.numel() for p in distrib_model.parameters())
        logging.info('total distributed parameters: '+str(tp_d)+'\n')
        tpt_d = sum(p.numel() for p in distrib_model.parameters() if p.requires_grad)
        logging.info('total distributed trainable parameters: '+str(tpt_d)+'\n')

# preprocess noise map (too ugly, fix later)
    sample = next(iter(train_loader))
    a,b = sample.inp.view(1, -1, *(sample.inp.size()[2:])).squeeze(0).float().shape[-2:]
    noise_map = noise_gen(a,b)

# resume state
    start_epoch = 1
    best_acc = np.Inf
    res_name='checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(program_dir+'/'+res_name, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            distrib_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if grank==0:
                logging.warning('restarting from #'+str(start_epoch)+' epoch!')
        except ValueError:
            if grank==0:
                logging.warning('restart file cannot be loaded, starting from 1st epoch!')

    only_test = start_epoch>args.epochs
    if grank==0 and only_test:
        logging.warning('given epochs are less than the one in the restart file!')
        logging.warning('only testing will be performed -- skipping training!')

# start trainin loop
    et = time.perf_counter()
    first_ep_t = last_ep_t = tot_ep_t = 0.0
    with open('out_loss.dat','a',encoding="utf-8") as outT:
        for epoch in range(start_epoch, args.epochs+1):
            # training
            loss_acc, train_t = train(distrib_model, train_sampler, loss_function, \
                device, train_loader, optimizer, epoch, scheduler_lr, noise_map)

            # save total/first/last epoch timer
            tot_ep_t += train_t
            if epoch == start_epoch:
                first_ep_t = train_t
            if epoch == args.epochs:
                last_ep_t = train_t

            # save state if found a better state
            is_best = loss_acc < best_acc
            if epoch % args.restart_int == 0 and not args.benchrun:
                save_state(epoch, distrib_model, loss_acc, optimizer, res_name, is_best)
                # reset best_acc
                best_acc = min(loss_acc, best_acc)

                # SAVE REGARDLESS A COPY with a different name for rewind
                save_state(epoch, distrib_model, loss_acc, optimizer, res_name+'_'+str(epoch), True)

            # write out loss and epoch
            if grank==0:
                outT.write("%4d   %5.15E\n" %(epoch, loss_acc))
                outT.flush()

            # empty cuda cache
            if args.cuda:
                torch.cuda.empty_cache()

# finalise training
    # save final state
    if not args.benchrun and not only_test:
        save_state(epoch, distrib_model, loss_acc, optimizer, res_name, True)

# debug final results
    if not only_test:
        debug_final(logging, start_epoch, epoch, first_ep_t, last_ep_t, tot_ep_t)

# start testing loop
    if not args.synt:
        test(distrib_model, loss_function, device, test_loader, noise_map)

# clean-up
    if grank==0:
        logging.info('final time: {:.2f}'.format(time.perf_counter()-st)+' s')
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    sys.exit()

# eof
