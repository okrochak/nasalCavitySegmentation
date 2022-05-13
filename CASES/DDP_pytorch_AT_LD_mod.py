#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version: 220318a

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
    parser = argparse.ArgumentParser(description='PyTorch actuated TBL')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')

    # model parsers
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--accum-iter', type=int, default=0,
                        help='accumulate gradient update (default: 1)')
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

    # optimizations
    parser.add_argument('--cudnn', action='store_true', default=False,
                        help='turn on cuDNN optimizations (default: False)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='turn on Automatic Mixed Precision (default: False)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs

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
def train(model, sampler, loss_function, device, train_loader, optimizer, epoch, grank, lt, scheduler):
    loss_acc = 0.0
    sampler.set_epoch(epoch)
    for batch_ndx, (samples) in enumerate(train_loader):
        do_backprop = ((batch_ndx + 1) % args.accum_iter == 0) or (batch_ndx + 1 == len(train_loader))
        inputs = samples.inp.view(1, -1, *(samples.inp.size()[2:])).squeeze(0).float().to(device)
        with torch.set_grad_enabled(True):
            if args.amp:
                with torch.cuda.amp.autocast():
                    # ===================forward=====================
                    predictions = model(inputs).float()
                    loss = loss_function(predictions, inputs) / args.accum_iter
                    # ===================backward====================
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
        #if batch_ndx % args.log_int == 0 and grank==0:
        if grank==0:
            print(f'Epoch: {epoch} / {100 * (batch_ndx + 1) / len(train_loader):3.2f}% complete',\
                    f' / {time.time() - lt:.2f} s / accumulated loss: {loss_acc}, back propagation: {do_backprop}')

    if args.schedule:
        scheduler.step()

    # profiling statistics
    if grank==0:
        print('TIMER: epoch time:', time.time()-lt, 's\n')

    return loss_acc

# test loop
def test(model, loss_function, device, test_loader, grank, gwsize):
    et = time.time()
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

    if grank==0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time()-et} s')
        if not args.testrun and not args.benchrun:
            plot_scatter(inputs[0][0].cpu().detach().numpy(),
                    predictions[0][0].cpu().detach().numpy(), 'test')

    # mean from gpus
    avg_test_loss = float(par_mean(test_loss,gwsize).cpu())
    avg_mean_sqr_diff = float(par_mean(mean_sqr_diff,gwsize).cpu())
    if grank==0:
        print(f'DEBUG: avg_test_loss: {avg_test_loss}')
        print(f'DEBUG: avg_mean_sqr_diff: {avg_mean_sqr_diff}\n')

    return avg_mean_sqr_diff

# encode export
def encode_exp(encode, device, train_loader, grank):
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
        print('DEBUG: args.accum_iter:',args.accum_iter)
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

        print('DEBUG: optimisation parsers:')
        print('DEBUG: args.cudnn:',args.cudnn)
        print('DEBUG: args.amp:',args.amp,'\n')

    # encapsulate the model on the GPU assigned to the current process
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
    # alt.
    #optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
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
                            device, train_loader, optimizer, epoch, grank, lt, scheduler_lr)
        else:
            loss_acc = train(distrib_model, train_sampler, loss_function, \
                            device, train_loader, optimizer, epoch, grank, lt, scheduler_lr)

        # save first/last epoch timer
        if epoch == start_epoch:
            first_ep_t = time.time()-lt
        if epoch == args.epochs-1:
            last_ep_t = time.time()-lt

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

        # empty cuda cache
        if args.cuda:
            torch.cuda.empty_cache()

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
        print(f'TIMER: last epoch time: {last_ep_t} s')
        print(f'TIMER: total epoch time: {time.time()-et} s')
        print(f'TIMER: average epoch time: {(time.time()-et)/args.epochs} s')
        if epoch > 1:
            print(f'TIMER: total epoch-1 time: {time.time()-et-first_ep_t} s')
            print(f'TIMER: average epoch-1 time: {(time.time()-et-first_ep_t)/(args.epochs-1)} s')
        if args.benchrun:
            print(f'TIMER: total epoch-2 time: {lt-et-first_ep_t} s')
            print(f'TIMER: average epoch-2 time: {(lt-et-first_ep_t)/(args.epochs-2)} s')
        print('DEBUG: memory req:',int(torch.cuda.max_memory_reserved(lrank)/1024/1024),'MB') \
                if args.cuda else 'DEBUG: memory req: - MB'
        print('DEBUG: memory summary:\n\n',torch.cuda.memory_summary(0)) if args.cuda else ''

# start testing loop
    avg_mean_sqr_diff = test(distrib_model, loss_function, device, test_loader, grank, gwsize)

# export first batch's latent space if needed (Turn to True)
    if False:
        encode = encoder().to(device)
        # distribute model to workers
        if args.cuda:
            distrib_encode = torch.nn.parallel.DistributedDataParallel(encode,\
                device_ids=[device], output_device=device)
        else:
            distrib_encode = torch.nn.parallel.DistributedDataParallel(encode)
        encode_exp(distrib_encode, device, train_loader, grank)

# clean-up
    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s')
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    sys.exit()

# eof
