#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to train a CAE model with large actuated TBL dataset
authors: RS, EI
version: 221021a
notes: modified by EI
"""

# std libs
import argparse, sys, platform, os, time, numpy as np, h5py, random, shutil, logging
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
    dtype = torch.float16 if args.amp else torch.float32
    data_u = torch.tensor(data_u,dtype=dtype).permute((1,0,2))[:-9,:,:]
    data_v = torch.tensor(data_v,dtype=dtype).permute((1,0,2))[:-9,:,:]
    data_w = torch.tensor(data_w,dtype=dtype).permute((1,0,2))[:-9,:,:]

    # normalise data to -1:1
    data_u = 2.0*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1.0
    data_v = 2.0*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1.0
    data_w = 2.0*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1.0

    return uniform_data(torch.cat((data_u, data_v, data_w)).view(( \
            3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3)))

# parsed settings
def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='PyTorch-DDP actuated TBL')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the'
                        ' local filesystem (default: ./)')
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')
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
    parser.add_argument('--export-latent', action='store_true', default=False,
                        help='export the latent space on testing for debug (default: False)')
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
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    # optimizations
    parser.add_argument('--cudnn', action='store_true', default=False,
                        help='turn on cuDNN optimizations (default: False)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='turn on Automatic Mixed Precision (default: False)')
    parser.add_argument('--scale-lr', action='store_true', default=False,
                        help='scale lr with #workers (default: False)')
    parser.add_argument('--accum-iter', type=int, default=1,
                        help='accumulate gradient update (default: 1 - turns off)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs

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
        logging.info('first epoch time: '+str(first_ep_t)+' s')
        logging.info('last epoch time: '+str(last_ep_t)+' s')
        logging.info('total epoch time: '+str(tot_ep_t)+' s')
        logging.info('average epoch time: '+str(tot_ep_t/done_epochs)+' s')
        if done_epochs>1:
            tot_ep_tm1 = tot_ep_t - first_ep_t
            logging.info('total epoch-1 time: '+str(tot_ep_tm1)+' s')
            logging.info('average epoch-1 time: '+str(tot_ep_tm1/(done_epochs-1))+' s')
        if args.benchrun and done_epochs>2:
            tot_ep_tm2 = tot_ep_t - first_ep_t - last_ep_t
            logging.info('total epoch-2 time: '+str(tot_ep_tm2)+' s')
            logging.info('average epoch-2 time: '+str(tot_ep_tm2/(done_epochs-2))+' s')
        # memory on worker 0
        if args.cuda:
            logging.info('memory req: '+str(int(torch.cuda.max_memory_reserved(0)/1024/1024))+' MB')
            logging.info('memory summary:\n'+str(torch.cuda.memory_summary(0)))

# network
class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

#Encoding layers - conv_en1 denotes 1st (1) convolutional layer for the encoding (en) part

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

#Decoding layers - conv_de1 denotes outermost (1) convolutional layer for the decoding (de) part

        self.conv5 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=0, bias=False)

    def forward(self, inp_x):
        """
        encoder - Convolutional layer is followed by a reLU activation which is then batch normalised
        """
# 1st encoding layer
        conv1 = self.leaky_reLU(self.bn1(self.conv1(inp_x)))
        conv2 = self.leaky_reLU(self.bn2(self.conv2(conv1)))
        conv3 = self.leaky_reLU(self.bn3(self.conv3(conv2)))
        conv4 = self.leaky_reLU(self.bn4(self.conv4(conv3)))
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
        conv5 = self.bn5(self.conv5(conv4))
        conv5 = interpolate(conv5, scale_factor=2, mode='bilinear', align_corners=True)
        conv5 = self.leaky_reLU(conv5)

        conv6 = self.leaky_reLU(self.bn6(self.conv6(conv5)))

        conv7 = self.bn7(self.conv7(conv6))
        conv7 = interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=True)
        conv7 = pad(conv7,(1,1,0,0)) if inp_x.shape[2]==250 or inp_x.shape[2]==450 \
                or inp_x.shape[2]==750 else pad(conv7,(1,1,1,1))
        conv7 = self.leaky_reLU(conv7)

        out_x = self.conv8(conv7)
        return out_x

# compression part - export latent space
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
    def forward(self, inp_x):
        conv1 = self.leaky_reLU(self.bn1(self.conv1(inp_x)))
        conv2 = self.leaky_reLU(self.bn2(self.conv2(conv1)))
        conv3 = self.leaky_reLU(self.bn3(self.conv3(conv2)))
        return self.leaky_reLU(self.bn4(self.conv4(conv3)))

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

# train loop
def train(model, sampler, loss_function, device, train_loader, optimizer, epoch, scheduler):
    # start a timer
    lt = time.perf_counter()

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
            schedule=torch.profiler.schedule(wait=1,warmup=1,active=args.epochs,repeat=1,skip_first=0),
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
    sampler.set_epoch(epoch)
    for batch_ndx, (samples) in enumerate(train_loader):
        do_backprop = ((batch_ndx + 1) % args.accum_iter == 0) or (batch_ndx + 1 == len(train_loader))
        inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float().to(device)
        with torch.set_grad_enabled(True):
            if args.amp:
                with torch.cuda.amp.autocast():
                    # forward pass
                    predictions = model(inputs).float()
                    loss = loss_function(predictions, inputs) / args.accum_iter
                    # backward pass
                    loss.backward()
                    if do_backprop:
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                predictions = model(inputs).float()
                loss = loss_function(predictions, inputs)
                loss.backward()
                if do_backprop:
                    optimizer.step()
                    optimizer.zero_grad()
        loss_acc+= loss.item()
        if batch_ndx % args.log_int == 0 and grank==0:
            print(f'Epoch: {epoch} / {100 * (batch_ndx + 1) / len(train_loader):06.2f}% done',\
                  f'/ loss: {loss_acc:19.16f}', end='')
            print(f' / bp: {do_backprop}') if not do_backprop else print(f'')

        # profiler step per batch
        if args.benchrun:
            prof.step()

    # lr scheduler
    if args.schedule:
        scheduler.step()

    # profiler end
    if args.benchrun:
        prof.stop()

    # timer for current epoch
    if grank==0:
        logging.info('epoch time: '+str(time.perf_counter()-lt)+' s\n')

    # printout profiler
    if args.benchrun and epoch==args.epochs-1 and grank==0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: benchmark of last epoch:\n')
        what1 = 'cuda' if args.cuda else 'cpu'
        print(prof.key_averages().table(sort_by='self_'+str(what1)+'_time_total', row_limit=-1))

    return loss_acc, time.perf_counter()-lt

# test loop
def test(model, loss_function, device, test_loader):
    et = time.perf_counter()
    model.eval()
    test_loss = 0.0
    mean_sqr_diff = 0.0
    with torch.no_grad():
        for count, (samples) in enumerate(test_loader):
            inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float().to(device)
            predictions = model(inputs).float()
            loss = loss_function(predictions, inputs) / args.accum_iter
            test_loss+= torch.nan_to_num(loss).item()/inputs.shape[0]
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            res = torch.mean(torch.square(torch.nan_to_num(predictions)-torch.nan_to_num(inputs)))
            res[torch.isinf(res)] = 0.0
            mean_sqr_diff = mean_sqr_diff*count/(count+1.0) + res.item()/(count+1.0)

    # timer
    if grank==0:
        logging.info('testing results:')
        logging.info('total testing time: '+str(time.perf_counter()-et)+' s')

    # mean from gpus
    avg_test_loss = float(par_mean(test_loss).cpu())
    avg_mean_sqr_diff = float(par_mean(mean_sqr_diff).cpu())
    if grank==0:
        logging.info('avg_test_loss: '+str(avg_test_loss))
        logging.info('avg_mean_sqr_diff: '+str(avg_mean_sqr_diff)+'\n')

    # plot comparison if needed
    if grank==0 and not args.skipplot and not args.testrun and not args.benchrun:
        plot_scatter(inputs[0][0].cpu().detach().numpy(),
                predictions[0][0].cpu().detach().numpy(), 'test')

# encode export
def encode_exp(encode, device, train_loader):
    for batch_ndx, (samples) in enumerate(train_loader):
        inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float().to(device)
        predictions = encode(inputs).float()

        # export the data
        ini = inputs.to('cpu').detach().numpy()
        res = predictions.to('cpu').detach().numpy()
        h5f = h5py.File('./test_'+str(batch_ndx)+'_'+str(grank)+'.h5', 'w')
        h5f.create_dataset('ini', data=ini)
        h5f.create_dataset('res', data=res)
        h5f.close()
        break

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
    model = autoencoder().to(device)

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
    optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr*lr_scale, weight_decay=args.wdecay)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    # alt.
    #optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr*lr_scale, weight_decay=args.wdecay)
    #scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

    # used lr and info on num. of parameters
    if grank==0:
        logging.info('current learning rate: '+str(args.lr*lr_scale)+'\n')
        #tp_d = sum(p.numel() for p in distrib_model.parameters())
        #logging.info('total distributed parameters: '+str(tp_d))
        tpt_d = sum(p.numel() for p in distrib_model.parameters() if p.requires_grad)
        logging.info('total distributed trainable parameters: '+str(tpt_d)+'\n')

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

# printout loss and epoch
    if grank==0 and not only_test:
        outT = open('out_loss.dat','w')

# start trainin loop
    et = time.perf_counter()
    tot_ep_t = 0.0
    for epoch in range(start_epoch, args.epochs+1):
        # training
        loss_acc, train_t = train(distrib_model, train_sampler, loss_function, \
            device, train_loader, optimizer, epoch, scheduler_lr)

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

        # write out loss and epoch
        if grank==0:
            outT.write("%4d   %5.15E\n" %(epoch, loss_acc))

        # empty cuda cache
        if args.cuda:
            torch.cuda.empty_cache()

    # close file
    if grank==0 and not only_test:
        outT.close()

# finalise training
    # save final state
    if not args.benchrun and not only_test:
        save_state(epoch, distrib_model, loss_acc, optimizer, res_name, True)

# debug final results
    if not only_test:
        debug_final(logging, start_epoch, epoch, first_ep_t, last_ep_t, tot_ep_t)

# start testing loop
    test(distrib_model, loss_function, device, test_loader)

# export first batch's latent space if needed (Turn to True)
    if args.export_latent:
        encode = encoder().to(device)
        # distribute model to workers
        if args.cuda:
            distrib_encode = torch.nn.parallel.DistributedDataParallel(encode,\
                device_ids=[device], output_device=device)
        else:
            distrib_encode = torch.nn.parallel.DistributedDataParallel(encode)
        encode_exp(distrib_encode, device, train_loader)

# clean-up
    if grank==0:
        logging.info('final time: '+str(time.perf_counter()-st)+' s')
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    sys.exit()

# eof
