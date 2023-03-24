#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version: 220422a

# std libs
import argparse, sys, platform, os, time, numpy as np, h5py, random, shutil, re

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate, pad
from torchvision import datasets, transforms

class custom_batch:
    def __init__(self, batch):
        imgs = [item[0][0].view((int(item[0][0].shape[0]/89), 89, item[0][0].shape[1], \
            item[0][0].shape[2], item[0][0].shape[3])) for item in batch]
        par_wid = [torch.Tensor([item[0][1]]).repeat(item[0][0].shape[0]) for item in batch]
        par_wl = [torch.Tensor([item[0][2]]).repeat(item[0][0].shape[0]) for item in batch]
        par_T = [torch.Tensor([item[0][3]]).repeat(item[0][0].shape[0]) for item in batch]
        par_amp = [torch.Tensor([item[0][4]]).repeat(item[0][0].shape[0]) for item in batch]
        par_cd = [torch.Tensor([item[0][5]]).repeat(item[0][0].shape[0]) for item in batch]
        par_Pnet = [torch.Tensor([item[0][6]]).repeat(item[0][0].shape[0]) for item in batch]

        self.imgs = torch.cat((imgs))
        self.par_wid = torch.cat(par_wid)
        self.par_wl = torch.cat(par_wl)
        self.par_T = torch.cat(par_T)
        self.par_amp = torch.cat(par_amp)
        self.par_cd = torch.cat(par_cd)
        self.par_Pnet = torch.cat(par_Pnet)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.imgs = self.imgs.pin_memory()
        self.par_wid = self.par_wid.pin_memory()
        self.par_wl = self.par_wl.pin_memory()
        self.par_T = self.par_T.pin_memory()
        self.par_amp = self.par_amp.pin_memory()
        self.par_cd = self.par_cd.pin_memory()
        self.par_Pnet = self.par_Pnet.pin_memory()
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

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    f1 = f[list(f.keys())[0]]

    cases_tbl = np.loadtxt(args.data_dir+'cases.dat', unpack = True)
    width_d = int(re.findall(r'\d+',path.split('/')[-6])[0])
    wavel_d = int(re.findall(r'\d+',path.split('/')[-5])[0])
    period_d = int(re.findall(r'\d+',path.split('/')[-4])[0])
    amp_d = int(re.findall(r'\d+',path.split('/')[-3])[0])
    for i in range(cases_tbl.shape[1]):
        if cases_tbl[1,i]==width_d and cases_tbl[2,i]==wavel_d and cases_tbl[3,i]==period_d and cases_tbl[4,i]==amp_d:
            c_d = 2*(cases_tbl[9,i]-cases_tbl[9,:].min())/(cases_tbl[9,:].max()-cases_tbl[9,:].min())-1
            Pnet = 2*(cases_tbl[12,i]-cases_tbl[12,:].min())/(cases_tbl[12,:].max()-cases_tbl[12,:].min())-1

    dtype = torch.float16 if args.amp else torch.float32
    data_u = torch.tensor(np.array(f1[list(f1.keys())[0]]['u']),dtype=dtype).permute((1,0,2))[:-9,:,:]
    data_u = 2*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1
    data_v = torch.tensor(np.array(f1[list(f1.keys())[0]]['v']),dtype=dtype).permute((1,0,2))[:-9,:,:]
    data_v = 2*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1
    data_w = torch.tensor(np.array(f1[list(f1.keys())[0]]['w']),dtype=dtype).permute((1,0,2))[:-9,:,:]
    data_w = 2*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1

    return uniform_data(torch.cat((data_u, data_v, data_w)).view(( \
            3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3))), \
            width_d/(cases_tbl[1,:].max()), \
            wavel_d/(cases_tbl[2,:].max()), \
            period_d/(cases_tbl[3,:].max()), \
            amp_d/(cases_tbl[4,:].max()), \
            c_d, \
            Pnet

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
    parser.add_argument('--log-int', type=int, default=5,
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
class nn_reg(nn.Module):
    def __init__(self):
        super(nn_reg, self).__init__()

        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(6144, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)
        self.fc5 = nn.Linear(14, 10)
        self.fc6 = nn.Linear(10, 2)

    def forward(self, inp_x, p1, p2, p3, p4):
        conv1 = self.leaky_reLU(self.bn1(self.conv1(inp_x)))
        conv2 = self.leaky_reLU(self.bn2(self.conv2(conv1)))
        conv3 = self.leaky_reLU(self.bn3(self.conv3(conv2)))
        conv4 = torch.flatten(self.leaky_reLU(self.bn4(self.conv4(conv3))), 1)
        fc1 = self.leaky_reLU(self.fc1(conv4))
        fc2 = self.leaky_reLU(self.fc2(fc1))
        fc3 = self.leaky_reLU(self.fc3(fc2))
        fc4 = self.leaky_reLU(self.fc4(fc3))
        fc4 = torch.cat((fc4, p1.unsqueeze(0).transpose(1,0), p2.unsqueeze(0).transpose(1,0), \
                p3.unsqueeze(0).transpose(1,0), p4.unsqueeze(0).transpose(1,0)),1)
        fc5 = self.leaky_reLU(self.fc5(fc4))
        return self.tanh(self.fc6(fc5))

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
def train(model, sampler, loss_function, device, train_loader, optimizer, epoch, grank, lt, scheduler_lr):
    loss_acc = 0.0
    sampler.set_epoch(epoch)
    for count, (inputs) in enumerate(train_loader):
        inps = inputs.imgs.view(1, -1, *(inputs.imgs.size()[2:])).squeeze(0).float().to(device)
        if args.amp:
            with torch.cuda.amp.autocast():
                # ===================forward=====================
                optimizer.zero_grad()
                # Forward pass
                predictions = model(inps, inputs.par_wid, inputs.par_wl, inputs.par_T, inputs.par_amp)
                # Compute loss function
                loss = loss_function(predictions.float(), \
                    torch.cat((inputs.par_cd.unsqueeze(0).float(), \
                    inputs.par_Pnet.unsqueeze(0).float())).transpose(1,0).to(device))
                # ===================backward====================
                loss.backward()  # Backward pass
                optimizer.step() # Optimizer step
        else:
            optimizer.zero_grad()
            predictions = model(inps, inputs.par_wid, inputs.par_wl, inputs.par_T, inputs.par_amp)
            loss = loss_function(predictions.float(), \
                torch.cat((inputs.par_cd.unsqueeze(0).float(), \
                inputs.par_Pnet.unsqueeze(0).float())).transpose(1,0).to(device))
            loss.backward()
            optimizer.step()

        loss_acc+= loss.item()
        if count % args.log_int == 0 and grank==0:
            print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):3.2f}% complete',\
                  f' / {time.time() - lt:.2f} s / accumulated loss: {loss_acc}')

    if args.schedule:
        scheduler_lr.step()

    # profiling statistics
    if grank==0:
        print('TIMER: epoch time:', time.time()-lt, 's\n')

    # debug results
    if grank==0:
        print('Inps:', torch.cat((inputs.par_cd.unsqueeze(0).float(),inputs.par_Pnet.unsqueeze(0).float())).transpose(1,0)[0])
        print('Preds:', predictions.float()[0])

    return loss_acc

# test loop
def test(model, loss_function, device, test_loader, grank, gwsize):
    et = time.time()
    model.eval()
    test_loss = 0.0
    count=0
    with torch.no_grad():
        for count, (inputs) in enumerate(test_loader):
            inps = inputs.imgs.view(1, -1, *(inputs.imgs.size()[2:])).squeeze(0).float().to(device)
            predictions = model(inps, inputs.par_wid, inputs.par_wl, inputs.par_T, inputs.par_amp)
            loss = loss_function(predictions.float(), \
                torch.cat((inputs.par_cd.unsqueeze(0).float(), \
                inputs.par_Pnet.unsqueeze(0).float())).transpose(1,0).to(device))
            test_loss+= torch.nan_to_num(loss).item()/inps.shape[0]

    if grank==0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time()-et} s')
        if not args.testrun or not args.benchrun:
            print(f'Prediction: {predictions[0,:].cpu().detach().numpy()}\n')
            print(f'Input: {torch.cat((inputs.par_cd.unsqueeze(0).float(),inputs.par_Pnet.unsqueeze(0).float())).transpose(1,0)[0,:]}\n')

    # mean from gpus
    avg_test_loss = float(par_mean(test_loss,gwsize).cpu())
    if grank==0:
        print(f'DEBUG: avg_test_loss: {avg_test_loss}')

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
    test_data = datasets.DatasetFolder(args.data_dir+'testfolder/Width1800',\
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
    model = nn_reg().to(device)

# distribute model to workers
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model,\
            device_ids=[device], output_device=device)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model)

# optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    #scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

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

        # save first epoch timer
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
    test(distrib_model, loss_function, device, test_loader, grank, gwsize)

# clean-up
    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s')
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    sys.exit()

# eof
