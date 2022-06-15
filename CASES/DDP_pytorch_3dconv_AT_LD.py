#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version: 210903a

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
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SequentialSampler

from torchnlp.samplers.distributed_sampler import DistributedSampler

class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.
    """
    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs
    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))
    def __len__(self):
        return len(self.batch_sampler)

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
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    f1 = f[list(f.keys())[0]]
    data_u = torch.FloatTensor(f1[list(f1.keys())[0]]['u']).permute((1,0,2))
    data_v = torch.FloatTensor(f1[list(f1.keys())[0]]['v']).permute((1,0,2))
    data_w = torch.FloatTensor(f1[list(f1.keys())[0]]['w']).permute((1,0,2))
    return torch.reshape(torch.cat((data_u, data_v, data_w)),
                    (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3))

# training settings
def pars_ini():
    parser = argparse.ArgumentParser(description='PyTorch AutoEncoder Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-int', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--concM', type=int, default=100, metavar='N',
                        help='conc MNIST to this factor (default: 100)')
    parser.add_argument('--backend', type=str, default='nccl', metavar='N',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--restart-int', type=int, default=10, metavar='N',
                        help='restart int per epoch (default: 10)')
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run (default: False)')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a test run (default: False)')
    return parser

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.reLU = nn.ReLU(True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.unpool = nn.MaxUnpool3d(kernel_size=2, stride=2, padding=0)
        self.tanhf = nn.Tanh()

#Encoding layers - conv_en1 denotes 1st (1) convolutional layer for the encoding (en) part
        self.conv_en1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm_en1 = nn.BatchNorm3d(32)
        self.conv_en2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bnorm_en2 = nn.BatchNorm3d(64)
        #self.conv_en3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=[2,1])
        #self.bnorm_en3 = nn.BatchNorm2d(64)

#Decoding layers - conv_de1 denotes outermost (1) convolutional layer for the decoding (de) part
        #self.bnorm_de3 = nn.BatchNorm2d(64)
        #self.conv_de3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=[0,1])
        self.bnorm_de2 = nn.BatchNorm3d(64)
        # for 3 layer compression, use the following for the 2nd deconvolution
        #self.conv_de2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=[0,1])
        # for 2 layer compression, use the following for the 2nd deconvolution
        self.conv_de2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=[1,1,1])
        self.bnorm_de1 = nn.BatchNorm3d(32)
        # for 3 layer compression, use the following for the 1st deconvolution
        #self.conv_de1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=[3,1])
        # for 2 layer compression, use the following for the 1st deconvolution
        self.conv_de1 = nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=[1,1,1])

    def forward(self, inp_x):
        """
        encoder - Convolutional layer is followed by a reLU activation which is then batch normalised
        """
# 1st encoding layer
        x = self.conv_en1(inp_x)
        x = self.leaky_reLU(x)
        x = self.bnorm_en1(x)
        x = self.pool(x)
# 2nd encoding layer
        x = self.conv_en2(x)
        x = self.leaky_reLU(x)
        x = self.bnorm_en2(x)
        x = self.pool(x)
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
        x = self.bnorm_de2(x)
        x = interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv_de2(x)
        x = self.leaky_reLU(x)
# 1st decoding layer
        x = self.bnorm_de1(x)
        x = interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv_de1(x)
        return self.tanhf(pad(x,(0,0,0,0,1,1))) if inp_x.shape[3]==300 or inp_x.shape[3]==400 else self.tanhf(pad(x,(0,0,1,1,1,1)))

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

# PARALLEL HELPERS
# sum of field over GPGPUs
def par_sum(field):
    res = torch.tensor(field).float().cuda()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    return res

# mean of field over GPGPUs
def par_mean(field,gwsize):
    res = torch.tensor(field).float().cuda()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    res/=gwsize
    return res

# max(field) over GPGPUs
def par_max(field):
    res = torch.tensor(field).float().cuda()
    dist.all_reduce(res,op=dist.ReduceOp.MAX,group=None,async_op=True).wait()
    return res

# min(field) over GPGPUs
def par_min(field):
    res = torch.tensor(field).float().cuda()
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
    res = torch.Tensor([field]).cuda()
    dist.reduce(res,dst=dest,op=oper,group=None,async_op=False)
    return res.item()

# gathers tensors from the whole group in a list (to all workers)
def par_allgather(field,gwsize):
    sen = torch.Tensor([field]).cuda()
    res = [torch.Tensor([field]).cuda() for i in range(gwsize)]
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
    parser = pars_ini()
    args = parser.parse_args()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

# initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend=args.backend)

    # get job rank info - rank==0 master gpu
    gwsize = dist.get_world_size()     # global world size - per run
    lwsize = torch.cuda.device_count() # local world size - per node
    grank = dist.get_rank()            # global rank - assign per run
    lrank = dist.get_rank()%lwsize     # local rank - assign per node


    # some debug
    if grank==0:
        print('TIMER: initialise:', time.time()-st, 's')
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:',sys.version)
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.lr:',args.lr)
        print('DEBUG: args.concM:',args.concM)
        print('DEBUG: args.backend:',args.backend)
        print('DEBUG: args.log_int:',args.log_int)
        print('DEBUG: args.restart_int:',args.restart_int)
        print('DEBUG: args.cuda:',args.cuda)
        print('DEBUG: args.testrun:',args.testrun)
        print('DEBUG: args.benchrun:',args.benchrun,'\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)

# load datasets
    turb_data = []

    transform = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    turb_data1 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1000',\
        loader=hdf5_loader, transform=transform, extensions='.hdf5')
    tb1_len = []
    tb1_len = [args.batch_size]*(len(turb_data1)//args.batch_size)
    tb1_len.append(len(turb_data1)%args.batch_size)
    turb_data1_spl = torch.utils.data.random_split(turb_data1, tb1_len)
    for i in range(len(turb_data1_spl)-1):
        turb_data.append(turb_data1_spl[i])

    turb_data2 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1200',\
        loader=hdf5_loader, transform=transform, extensions='.hdf5')
    tb2_len = []
    tb2_len = [args.batch_size]*(len(turb_data2)//args.batch_size)
    tb2_len.append(len(turb_data2)%args.batch_size)
    turb_data2_spl = torch.utils.data.random_split(turb_data2, tb2_len)
    for i in range(len(turb_data2_spl)-1):
        turb_data.append(turb_data2_spl[i])

    #turb_data3 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1600',\
    #    loader=hdf5_loader, transform=transform, extensions='.hdf5')
    #tb3_len = []
    #tb3_len = [args.batch_size]*(len(turb_data3)//args.batch_size)
    #tb3_len.append(len(turb_data3)%args.batch_size)
    #turb_data3_spl = torch.utils.data.random_split(turb_data3, tb3_len)
    #for i in range(len(turb_data3_spl)-1):
    #    turb_data.append(turb_data3_spl[i])

    #turb_data4 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width1800',\
    #    loader=hdf5_loader, transform=transform, extensions='.hdf5')
    #tb4_len = []
    #tb4_len = [args.batch_size]*(len(turb_data4)//args.batch_size)
    #tb4_len.append(len(turb_data4)%args.batch_size)
    #turb_data4_spl = torch.utils.data.random_split(turb_data4, tb4_len)
    #for i in range(len(turb_data4_spl)-1):
    #    turb_data.append(turb_data4_spl[i])

    #turb_data5 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width3000',\
    #    loader=hdf5_loader, transform=transform, extensions='.hdf5')
    #tb5_len = []
    #tb5_len = [args.batch_size]*(len(turb_data5)//args.batch_size)
    #tb5_len.append(len(turb_data5)%args.batch_size)
    #turb_data5_spl = torch.utils.data.random_split(turb_data5, tb5_len)
    #for i in range(len(turb_data5_spl)-1):
    #    turb_data.append(turb_data5_spl[i])

    turbdata_cat = torch.utils.data.ConcatDataset(turb_data)

    test_data = []

    test_data1 = datasets.DatasetFolder(args.data_dir+'testfolder/Width1000',\
         loader=hdf5_loader, transform=transform, extensions='.hdf5')
    tb1_len = []
    tb1_len = [args.batch_size]*(len(test_data1)//args.batch_size)
    tb1_len.append(len(test_data1)%args.batch_size)
    test_data1_spl = torch.utils.data.random_split(test_data1, tb1_len)
    for i in range(len(test_data1_spl)-1):
        test_data.append(test_data1_spl[i])

    test_data2 = datasets.DatasetFolder(args.data_dir+'testfolder/Width1200',\
        loader=hdf5_loader, transform=transform, extensions='.hdf5')
    tb2_len = []
    tb2_len = [args.batch_size]*(len(test_data2)//args.batch_size)
    tb2_len.append(len(test_data2)%args.batch_size)
    test_data2_spl = torch.utils.data.random_split(test_data2, tb2_len)
    for i in range(len(test_data2_spl)-1):
        test_data.append(test_data2_spl[i])

    #test_data3 = datasets.DatasetFolder(args.data_dir+'testfolder/Width1600',\
    #    loader=hdf5_loader, transform=transform, extensions='.hdf5')
    #tb3_len = []
    #tb3_len = [args.batch_size]*(len(test_data3)//args.batch_size)
    #tb3_len.append(len(test_data3)%args.batch_size)
    #test_data3_spl = torch.utils.data.random_split(test_data3, tb3_len)
    #for i in range(len(test_data3_spl)-1):
    #    test_data.append(test_data3_spl[i])

    #test_data4 = datasets.DatasetFolder(args.data_dir+'testfolder/Width1800',\
    #    loader=hdf5_loader, transform=transform, extensions='.hdf5')
    #tb4_len = []
    #tb4_len = [args.batch_size]*(len(test_data4)//args.batch_size)
    #tb4_len.append(len(test_data4)%args.batch_size)
    #test_data4_spl = torch.utils.data.random_split(test_data4, tb4_len)
    #for i in range(len(test_data4_spl)-1):
    #    test_data.append(test_data4_spl[i])

    #test_data5 = datasets.DatasetFolder(args.data_dir+'testfolder/Width3000',\
    #    loader=hdf5_loader, transform=transform, extensions='.hdf5')
    #tb5_len = []
    #tb5_len = [args.batch_size]*(len(test_data5)//args.batch_size)
    #tb5_len.append(len(test_data5)%args.batch_size)
    #test_data5_spl = torch.utils.data.random_split(test_data5, tb5_len)
    #for i in range(len(test_data5_spl)-1):
    #    test_data.append(test_data5_spl[i])

    testdata_cat = torch.utils.data.ConcatDataset(test_data)

    # restricts data loading to a subset of the dataset exclusive to the current process
    btrain_sampler = BatchSampler(SequentialSampler(turbdata_cat), batch_size=args.batch_size, drop_last=False)
    btest_sampler = BatchSampler(SequentialSampler(testdata_cat), batch_size=args.batch_size, drop_last=False)

    train_sampler = DistributedBatchSampler(btrain_sampler, num_replicas=gwsize, rank=grank)
    test_sampler = DistributedBatchSampler(btest_sampler, num_replicas=gwsize, rank=grank)

# distribute dataset to workers
    train_loader = torch.utils.data.DataLoader(turbdata_cat, batch_sampler=train_sampler, num_workers=0, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testdata_cat,
        batch_sampler=test_sampler, num_workers=0, pin_memory=True, shuffle=False)

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
    optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr, weight_decay=0.003)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# resume state
    start_epoch = 0
    best_acc = np.Inf
    res_name='checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank}
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
        loss_acc = 0.0
        count=0
        #train_sampler.set_epoch(epoch)
        for inputs, labels in train_loader:
            inputs = inputs.permute(0,2,1,3,4).to(device)
            # ===================forward=====================
            optimizer.zero_grad()
            predictions = distrib_model(inputs)         # Forward pass
            loss = loss_function(predictions.float(), inputs.float())   # Compute loss function
            # ===================backward====================
            loss.backward()                                        # Backward pass
            optimizer.step()                                       # Optimizer step
            loss_acc+= loss.item()
            if count % args.log_int == 0 and grank==0 and lrank==0:
                print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):.2f}% complete',\
                      f' / {time.time() - lt:.2f} s / accumulated loss: {loss_acc}\n')
            count+=1

        scheduler_lr.step()

        # save state if found a better state
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0 and not args.benchrun:
            save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,is_best)
            # reset best_acc
            best_acc = min(loss_acc, best_acc)

        if grank==0:
            print('TIMER: epoch time:', time.time()-lt, 's')

# finalise training
    # save final state
    if not args.benchrun:
        save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,True)
    dist.barrier()

    # some debug
    if grank==0:
        print(f'\n--------------------------------------------------------')
        print('DEBUG: training results:\n')
        print(f'TIMER: last epoch time: {time.time()-lt} s\n')
        print(f'TIMER: total epoch time: {time.time()-et} s\n')
        if args.cuda:
            print('DEBUG: memory req:',int(torch.cuda.memory_reserved(lrank)/1024/1024),'MB')
        elif not args.cuda:
            print('DEBUG: memory req: - MB')

# start testing loop
    et = time.time()
    distrib_model.eval()
    test_loss = 0.0
    mean_sqr_diff = []
    count=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.permute(0,2,1,3,4).to(device)
            predictions = distrib_model(inputs)
            loss = loss_function(predictions.float(), inputs.float())
            test_loss+= loss.item()/inputs.shape[0]
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            mean_sqr_diff.append(\
                torch.mean(torch.square(predictions.float()-inputs.float())).item())
            count+=1

    # mean from dataset (ignore if just 1 dataset)
    if count>1:
        mean_sqr_diff=np.mean(mean_sqr_diff)
    if grank==0:
        print(f'TIMER: total testing time: {time.time()-et} s\n')
        if args.testrun:
            plot_scatter(inputs[0][0][0].cpu().detach().numpy(),
                    predictions[0][0][0].cpu().detach().numpy(), 'test')

# finalise testing
    # mean from gpus
    avg_mean_sqr_diff = par_mean(mean_sqr_diff,gwsize)
    if grank==0:
        print(f'DEBUG: avg_mean_sqr_diff: {avg_mean_sqr_diff}\n')

# clean-up
    dist.destroy_process_group()

    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s\n')

if __name__ == "__main__":
    main()
    sys.exit()

# eof


