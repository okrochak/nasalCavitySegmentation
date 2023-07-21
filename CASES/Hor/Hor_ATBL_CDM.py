#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to train custom diffusion model with large actuated TBL dataset.
instead of noise, dataset is filtered with Gaussian filter and artificial turbulence is added.
uses PINNs in loss with 3D conv
authors: EI, RS
version: 230721a
notes: 
option to use a cubic extraction for the training, see --cube argument for details!
network architecture is moved to CDM_network.py
for cost Perlin noise is selected, use spectral methods later on.
works for non-actuated case at this moment!
"""

# std libs
import argparse, sys, platform, os, time, h5py, random, shutil, logging
import matplotlib.pyplot as plt, numpy as np, scipy as sp

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
import horovod.torch as hvd
from torchvision import datasets, transforms
#from CDM_network import U_Net, AttU_Net, R2AttU_Net, cdm_2d, noise_gen_2d # 2D Conv
from CDM_network import TDU_Net, Att3DU_Net, R2Att3DU_Net, cdm_3d, noise_gen_3d # 3D Conv
#from lion_pytorch import Lion # TEST

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
    parser.add_argument('--cube', action='store_true', default=False,
                        help='cut cubes that is being cut from the sample instead of full field (default: False)')
    parser.add_argument('--cubeD', type=int, default=16,
                        help='size of the cube that is being cut from the sample (default: 16)')
    parser.add_argument('--cubeM', type=int, default=20,
                        help='increase #cubes with this factor (default: 20)')
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
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer,'
                        'note: Horovod only! (default: 1.0)')

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
    parser.add_argument('--use-fork', action='store_true', default=False,
                        help='use forkserver for dataloading,'
                        'note: Horovod only! + problems with IB (default: False)')

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
                        help='turn on Automatic Mixed Precision - accuracy issues! (default: False)')
    parser.add_argument('--reduce-prec', action='store_true', default=False,
                        help='reduce precision of the dataset for faster I/O (default: False)')
    parser.add_argument('--scale-lr', action='store_true', default=False,
                        help='scale lr with #workers (default: False)')
    parser.add_argument('--sigma-lr', action='store_true', default=False,
                        help='scale lr with sigma in CDM (default: False)')
    parser.add_argument('--accum-iter', type=int, default=1,
                        help='accumulate gradient update (default: 1 - turns off)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce,'
                        'note: Horovod only! (default: False)')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction,'
                        'ignores scale-lr option, note: Horovod only (default: false)')
    parser.add_argument('--batch-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                        'executing allreduce across workers; it multiplies '
                        'total batch size, note: Horovod only (default: 1 - turns off)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs

# postproc
def plot_scatter(org_img, inp_img, out_img, epoch, sigma, final=False):
    ns = int(org_img.shape[0]/2) # extract middle plane
    fig = plt.figure(figsize = (4,12))
    plt.rc('font', size=10)
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(org_img[ns,:,:],interpolation='None')
    ax1.set_title('Original')
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(inp_img[ns,:,:], interpolation='None')
    ax2.set_title('Input')
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(out_img[ns,:,:], interpolation='None')
    ax3.set_title('Output')

    outName = 'vfield_recon_CDM_train_'+str(epoch)+'_'+str(sigma)
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
        a = args.cubeD if args.cube else 89
        concat = args.cubeM if args.cube else 0
        if args.cube:
            inp = [item[0].reshape(int(item[0].shape[0]/a)*(concat+1), a, 3, a, a) for item in batch]
        else:
            inp = [item[0].reshape(int(item[0].shape[0]/a), a, item[0].shape[1], \
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

    if args.cube:
        # cut a**3 cube from data for sampling
        a = args.cubeD # cube dims
        concat = args.cubeM # #random cubes
        try:
            assert a<data_u.size()[0] and a<data_u.size()[1] and a<data_u.size()[2]
        except AssertionError:
            print(f'box is too large, larger than the domain size!')
            sys.exit()
        # randomly cut boxes from the domain and stack them together
        cut_u = torch.ones((a,a,a))
        cut_v = torch.ones((a,a,a))
        cut_w = torch.ones((a,a,a))
        stack_u = torch.clone(cut_u)
        stack_v = torch.clone(cut_v)
        stack_w = torch.clone(cut_w)
        for c in range(concat):
            # random numbers, not exceeding domain length
            n = random.randint(0, data_u.size()[0]-a)
            m = random.randint(0, data_u.size()[1]-a)
            j = random.randint(0, data_u.size()[2]-a)
            # cut
            cut_u = data_u[n:n+a,m:m+a,j:j+a]
            cut_v = data_v[n:n+a,m:m+a,j:j+a]
            cut_w = data_w[n:n+a,m:m+a,j:j+a]
            # stack
            stack_u = torch.cat((stack_u,cut_u))
            stack_v = torch.cat((stack_v,cut_v))
            stack_w = torch.cat((stack_w,cut_w))

        return torch.cat((stack_u, stack_v, stack_w)).view((3*(concat+1),a,a,a)).permute((1,0,2,3))
    else:
        # use full field 
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
        logging.info('local ranks: '+str(lwsize)+' / global ranks: '+str(gwsize))
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

"""loss defined as
https://www.sciencedirect.com/science/article/pii/S1540748920300481
originally for GANs
Loss = b1*L1_adversarial + b2*L2_pixel + b3*L3_gradient + b4*L_physics
L1_adversial = 0 as no GANs here
"""
# custom loss function
class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # constants and specing
        self.b2 = 0.88994
        self.b3 = 0.06
        self.b4 = 0.05
        self.del_x = 0.2598e-3
        self.del_y = torch.tensor(np.loadtxt(args.data_dir+'coords_y.dat')[:-9]*1e-3,\
                dtype=torch.float32).to(device)
        self.del_z = 0.0866e-3

    #def forward(self, inputs, targets):
    #    # loss for 2d convolution based on pixels
    #    return torch.mean((inputs-targets)**2.0)

    def forward(self, inputs, targets):
        if not args.cube:
            # loss pixel
            L_2 = self.b2*torch.mean((inputs-targets)**2.0)

            # loss gradient
            # compute 3 gradients of inputs and targets
            ix = torch.gradient(inputs, spacing=self.del_x,dim=4)[0]
            iy = torch.gradient(inputs, spacing=(self.del_y,),dim=2)[0]
            iz = torch.gradient(inputs, spacing=self.del_z,dim=3)[0]
            tx = torch.gradient(targets,spacing=self.del_x,dim=4)[0]
            ty = torch.gradient(targets,spacing=(self.del_y,),dim=2)[0]
            tz = torch.gradient(targets,spacing=self.del_z,dim=3)[0]
            # MSE of gradient of the inputs to targets
            L_3  = self.b3*torch.mean((ix-tx)**2.0)
            L_3 += self.b3*torch.mean((iy-ty)**2.0)
            L_3 += self.b3*torch.mean((iz-tz)**2.0)

            # loss physics - compressibility (need 3D conv)
            # dudx + dvdy + dwdz != 0
            L_4 = self.b4*torch.mean((tx[:,0,:,:,:] + ty[:,1,:,:,:] + tz[:,2,:,:,:]).abs())

            # normalise L3-L4 wrt L2 so each L are in the same magnitude
            L_3 = L_3 / 10.0**torch.ceil(torch.log10(L_3/L_2))
            L_4 = L_4 / 10.0**torch.ceil(torch.log10(L_4/L_2))

            return L_2+L_3+L_4
        else:
            # cube cut does not have a simple way to get del_y, so it is removed for this iteration
            return torch.mean((inputs-targets)**2.0)

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
        # 2D conv
        #inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float()
        # 3D conv
        inputs = samples.inp.permute(0,2,1,3,4).float()

        # diffusion model
        lt_3 = time.perf_counter()
        data = cdm_3d(inputs, noise_map, epoch)
        lt_2 += time.perf_counter() - lt_3

        # adjust LR with sigma just testing!
        if args.sigma_lr:
            adjust_learning_rate(optimizer, epoch, data.sigma)

        # train part
        with torch.set_grad_enabled(True):
            if args.amp:
                with torch.cuda.amp.autocast():
                    # forward pass
                    predictions = model(data.inputs_cdm.to(device)).float()
                    loss = loss_function(predictions, inputs.to(device)) / args.accum_iter
                    # backward pass
                    loss.backward()
                    if do_backprop:
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                predictions = model(data.inputs_cdm.to(device)).float()
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
                data.inputs_cdm[0][0].cpu().detach().numpy(), \
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
                # 2D conv
                #inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float().to(device)
                # 3D conv
                inputs = samples.inp.permute(0,2,1,3,4).float().to(device)

                # test highly diffused image for diffusion model
                data = cdm_3d(inputs.cpu(), noise_map, sigma=sigma_test)
                predictions = model(data.inputs_cdm.to(device)).float()
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
                    data.inputs_cdm[0][0].cpu().detach().numpy(), \
                    predictions[0][0].cpu().detach().numpy(), \
                    epoch=args.epochs, sigma=sigma_test, final=True)

    if grank==0:
        logging.info('total testing time: {:.2f}'.format(time.perf_counter()-et)+' s')

def adjust_learning_rate(optimizer, sigma, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr*factor

# PARALLEL HELPERS
# sum of field over GPGPUs
def par_sum(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    hvd.allreduce_(res, average=False)
    return res

# mean of field over GPGPUs
def par_mean(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
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
    pars_ini()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # get directory
    program_dir = os.getcwd()

    # start the time for profiling
    st = time.perf_counter()

# initializes the distributed backend which will take care of sychronizing nodes/GPUs
    hvd.init()

# deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

# get job rank info -- rank==0 master gpu
    global device, lwsize, gwsize, grank, lrank
    gwsize = hvd.size()       # global world size - per run
    lwsize = hvd.local_size() # local world size - per node
    grank =  hvd.rank()       # global rank - assign per run
    lrank =  hvd.local_rank() # local rank - assign per node

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
    #distrib_model = TDU_Net().to(device) # 3D U-Net
    distrib_model = Att3DU_Net().to(device) # Attention 3D U-Net
    #distrib_model = R2Att3DU_Net().to(device) #Residual Recurrent Attention 3D U-Net, !memory problems!

    # scale lr with #workers
    lr_scale = gwsize if args.scale_lr else 1

    # By default, Adasum doesn't need scaling up learning rate.
    if args.cuda:
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        lr_scale = lwsize if args.use_adasum else lr_scale

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# optimizer
    loss_function = nn.MSELoss()
    #optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr*lr_scale, weight_decay=args.wdecay)
    optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr*lr_scale)
    #scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000)

# distribute model to workers
    # Horovod: broadcast parameters & optimizer state
    hvd.broadcast_parameters(distrib_model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer, \
                                named_parameters=distrib_model.named_parameters(), \
                                op=hvd.Adasum if args.use_adasum else hvd.Average, \
                                compression=compression, \
                                gradient_predivide_factor=args.gradient_predivide_factor, \
                                backward_passes_per_step=args.batch_per_allreduce)

    # used lr and info on num. of parameters
    if grank==0:
        logging.info('current learning rate: '+str(args.lr*lr_scale)+'\n')
        tp_d = sum(p.numel() for p in distrib_model.parameters())
        logging.info('total distributed parameters: '+str(tp_d)+'\n')
        tpt_d = sum(p.numel() for p in distrib_model.parameters() if p.requires_grad)
        logging.info('total distributed trainable parameters: '+str(tpt_d)+'\n')

# preprocess noise map (too ugly, fix later)
    et = time.perf_counter()
    res_name='noise_map.h5'
    if os.path.isfile(res_name):
        # read noise_map
        with h5py.File(res_name, 'r') as h5f:
            a_group_key = list(h5f.keys())[0]
            noise_map = h5f[a_group_key][()]
    else:
        # generate noise_map if not existent (2d conv)
        #sample = next(iter(train_loader))
        #a,b = sample.inp.view(1, -1, *(sample.inp.size()[2:])).squeeze(0).float().shape[-2:]
        #noise_map = noise_gen(a,b)
        # generate noise_map if not existent (3d conv - expensive)
        sample = next(iter(train_loader))
        a,b,c = sample.inp.permute(0,2,1,3,4).float().shape[2::]
        noise_map = noise_gen_3d(a,b,c)
        if grank == 0: 
            with h5py.File(res_name, 'w') as h5f:
                h5f.create_dataset(res_name, data=noise_map)
    logging.info('Noise read in: {:.2f}'.format(time.perf_counter()-et)+' s\n')

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
    hvd.shutdown()

if __name__ == "__main__":
    main()
    sys.exit()

# eof
