#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: RS, EI
# version: 211029a

# std libs
import argparse, sys, platform, os, time, numpy as np, h5py, random, shutil, re, math
import matplotlib.pyplot as plt

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate, pad
from torchvision import datasets, transforms

from CustomDistSampler import DistributedCustomSampler

import utils

#from CDM_network import U_Net

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

# plot reconstruction
def plot_line(inp_img, out_img, data_org):
    plt.plot(inp_img,label='target')
    plt.plot(out_img,label='prediction')
    plt.legend()
    plt.savefig('TR_CDM_recon_'+data_org+str(random.randint(0,100))+'.pdf')
    plt.close()

def collate_batch(batch):
    imgs = [item[0].view(int(item[0].shape[0]/88), 88, item[0].shape[1], \
        item[0].shape[2], item[0].shape[3]) for item in batch]
    targets = [item[1] for item in batch]
    imgs = torch.cat((imgs))
    return imgs, targets

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

# loader for turbulence HDF5 data
def hdf5_loader(path):
    f = h5py.File(path, 'r')
    data = torch.from_numpy(np.array(f['latvar']))
    data = 2*(data-torch.min(data))/(torch.max(data)-torch.min(data))-1
    return data
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
    #f1 = f[list(f.keys())[0]]
    #data_u = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['u'])).permute((1,0,2))[:-9,:,:]
    #data_u = 2*(data_u-torch.min(data_u))/(torch.max(data_u)-torch.min(data_u))-1
    #data_v = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['v'])).permute((1,0,2))[:-9,:,:]
    #data_v = 2*(data_v-torch.min(data_v))/(torch.max(data_v)-torch.min(data_v))-1
    #data_w = torch.from_numpy(np.array(f1[list(f1.keys())[0]]['w'])).permute((1,0,2))[:-9,:,:]
    #data_w = 2*(data_w-torch.min(data_w))/(torch.max(data_w)-torch.min(data_w))-1
    #return uniform_data(torch.reshape(torch.cat((data_u, data_v, data_w)),
    #           (3,data_u.shape[0],data_u.shape[1],data_u.shape[2])).permute((1,0,2,3)))

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
    parser.add_argument('--schedule', action='store_true', default=True,
                        help='enable scheduler in the training (default: False)')

    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=1,
                        help='log interval per training')
    parser.add_argument('--save_hdf5', action='store_true', default=False,
                        help='save test input and predictions to hdf5 (default: False)')

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

def get_src_trg(
        seq, 
        encseq_l, 
        tgtseq_l):
        assert len(seq) == encseq_l + tgtseq_l, "Sequence length does not equal (input length + target length)"
        # encoder input
        src = seq[:encseq_l]         
        tgt = seq[encseq_l-1:len(seq)-1]
        #tgt = tgt[:, 0]
        if len(tgt.shape) == 1:
            tgt = tgt.unsqueeze(-1)
        assert len(tgt) == tgtseq_l, "Length of trg does not match target sequence length"
        tgt_y = seq[-tgtseq_l:]
        #tgt_y = tgt_y[:, 0]
        assert len(tgt_y) == tgtseq_l, "Length of trg_y does not match target sequence length"
        return src, tgt, tgt_y 


def generate_square_mask(dim1, dim2):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

##Based on https://github.com/KasperGroesLudvigsen/influenza_transformer
class PositionalEncoder(nn.Module):
    def __init__(self, 
        dropout: float=0.1, 
        maxseq_l: int=5000, 
        d_model: int=512,
        batch_first: bool=False):
        super(PositionalEncoder, self).__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        position = torch.arange(maxseq_l).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(maxseq_l, 1, d_model)        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)

##Based on https://github.com/KasperGroesLudvigsen/influenza_transformer
class transformer_tbl(nn.Module):
    def __init__(self, 
        inp_dim: int,
        batch_first: bool,
        num_pred_features: int,
        maxseq_l: int=5000,
        dim_val: int=2048,  
        n_enc_lay: int=6,
        n_dec_lay: int=6,
        n_heads: int=8,
        dropout_enc: float=0.2, 
        dropout_dec: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_ff_enc: int=4096,
        dim_ff_dec: int=4096):

        super(transformer_tbl, self).__init__()
        self.enc_inp = nn.Linear(inp_dim, dim_val)
        self.dec_inp = nn.Linear(num_pred_features, dim_val)  
        self.linear_map = nn.Linear(dim_val, num_pred_features)
        self.tanh = nn.Tanh()

        # Positional Encoder
        self.positional_enc_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_ff_enc,
            dropout=dropout_enc,
            batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=n_enc_lay, 
            norm=None)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_ff_dec,
            dropout=dropout_dec,
            batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=dec_layer,
            num_layers=n_dec_lay, 
            norm=None)

    def forward(self, src, tgt, src_mask, 
                tgt_mask):
        src = self.enc_inp(src) 
        src = self.positional_enc_layer(src) 
        src = self.encoder(src=src)
        decoder_output = self.dec_inp(tgt) 
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask)

        decoder_output = self.tanh(self.linear_map(decoder_output)) 

        return decoder_output


def run_encoder_decoder_inference(model, src, forecast_window, batch_size, device, batch_first):

    target_seq_dim = 0 if batch_first == False else 1
    tgt = src[-1, :, :] if batch_first == False else src[:, -1, :] # shape [1, batch_size, 1]

    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0).unsqueeze(0) # change from [1] to [1, 1, 1]

    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0)

    for _ in range(forecast_window-1):
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]
        dim_b = src.shape[1] if batch_first == True else src.shape[0]
        tgt_mask = utils.generate_square_subsequent_mask(dim1=dim_a, dim2=dim_a)
        src_mask = utils.generate_square_subsequent_mask(dim1=dim_a, dim2=dim_b)
        
        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask) 

        if batch_first == False:
            last_predicted_value = prediction[-1, :, :] 
            last_predicted_value = last_predicted_value.unsqueeze(0)
        else:
            last_predicted_value = prediction[:, -1, :]
            last_predicted_value = last_predicted_value.unsqueeze(-1)

        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)

    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]
    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    tgt_mask = utils.generate_square_subsequent_mask(dim1=dim_a, dim2=dim_a)
    src_mask = utils.generate_square_subsequent_mask(dim1=dim_a, dim2=dim_b)

    final_prediction = model(src, tgt, src_mask, tgt_mask)

    return final_prediction

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
            torch.save(state,res_name)
            print(f'DEBUG: state in {grank} is saved on epoch:{epoch} in {time.time()-rt} s')

# deterministic dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_to_hdf5(inps, preds, preds_QoI, width, wavelength, period, amplitude, inp_Cd, inp_Pnet, count, grank):
    hf5 = h5py.File('/p/scratch/raise-ctp1/T31_LD_CAE_recon/2DCAEregr/test_w1800/TBL_CAErecon'+str(count)+'_'+str(random.randint(0,100000))+'.h5', 'w')
    g1 = hf5.create_group(str(grank))
    #g1.create_dataset('grank/inputs_NN', data=inps.cpu())
    #g1.create_dataset('grank/predictions_NN', data=preds.cpu())
    #g1.create_dataset('grank/param_width', data=width.cpu())
    #g1.create_dataset('grank/param_wavelength', data=wavelength.cpu())
    #g1.create_dataset('grank/param_period', data=period.cpu())
    #g1.create_dataset('grank/param_amplitude', data=amplitude.cpu())
    g1.create_dataset('grank/param_inp_Cd', data=inp_Cd.cpu())
    g1.create_dataset('grank/param_inp_Pnet', data=inp_Pnet.cpu())
    g1.create_dataset('grank/preds_QoI', data=preds_QoI.cpu())
    dist.barrier()
    if grank==0:
        hf5.close()

# train loop
def train(model, sampler, loss_function, device, train_loader, optimizer, epoch, grank, lt, scheduler_lr):
    loss_acc = 0.0
    count=0
    #sampler.set_epoch(epoch)
    for inputs, labels in train_loader:
        #inps = inputs[0].float().to(device)
        #inps = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0).float().to(device)
        inps = inputs.view(*inputs.size()[:-4], -1).float().to(device)
        #inps = torch.mean(inputs,dim=[3,4]).float().to(device)
        
        encseq_l = len(inps)-2
        outseq_l = 2
        tgtseq_l = outseq_l

        src,tgt,tgt_y = get_src_trg(inps, encseq_l, tgtseq_l)
        tgt_mask = generate_square_mask(dim1=outseq_l,dim2=outseq_l)
        src_mask = generate_square_mask(dim1=outseq_l,dim2=encseq_l)
        
        # ===================forward=====================
        optimizer.zero_grad()
        predictions = model(src,tgt,src_mask,tgt_mask)         # Forward pass
        loss = loss_function(predictions.float(), tgt_y.float())   # Compute loss function
        # ===================backward====================
        loss.backward()                                        # Backward pass
        optimizer.step()                                       # Optimizer step
        loss_acc+= loss.item()
        if count % args.log_int == 0 and grank==0:
            print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):3.2f}% complete',\
                  f' / {time.time() - lt:.2f} s / accumulated loss: {loss_acc}')
            #print(f'Predictions: {predictions.float()}, Target: {tgt_y.float()}')
        count+=1

    if args.schedule:
        scheduler_lr.step()

    # profiling statistics
    if grank==0:
        #print(len(tgt_y.float().flatten().cpu().detach().numpy()))
        #print(predictions.float().flatten().cpu().detach().numpy().shape())
        print('TIMER: epoch time:', time.time()-lt, 's\n')
        #plot_line(tgt_y.float().flatten().cpu().detach().numpy()[:20], predictions.float().flatten().cpu().detach().numpy()[:20], 'train')

    return loss_acc

# test loop
def test(model, loss_function, device, test_loader, grank, gwsize):
    et = time.time()
    model.eval()
    test_loss = 0.0
    mean_sqr_diff = []
    count=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            #inputs = inputs.view(1, -1, *(inputs.size()[2:])).squeeze(0).float().to(device)
            inps = inputs.view(*inputs.size()[:-4], -1).float().to(device)
            #inps = torch.mean(inputs,dim=[3,4]).float().to(device)

            #encseq_l = len(inps)-2
            #outseq_l = 2
            src = inps
            forecast_window = 2
            tgtseq_l = forecast_window

            #src,tgt,tgt_y = get_src_trg(inps, encseq_l, tgtseq_l)

            predictions = run_encoder_decoder_inference(
                model=model, 
                src=src, 
                forecast_window=forecast_window,
                batch_size=src.shape[1],
                device=device,
                batch_first=False)

            tgt_y = src[-tgtseq_l:]

            loss = loss_function(predictions.float(), tgt_y.float())
            test_loss+= loss.item()/inps.shape[0]
            
            # mean squared prediction difference (Jin et al., PoF 30, 2018, Eq. 7)
            #mean_sqr_diff.append(\
            #    torch.mean(torch.square(predictions.float()-tgt_y.float())).item())
            #print(f'Predictions: {predictions.float()}, Target: {tgt_y.float()}')
            mean_sqr_diff.append(\
                torch.mean(torch.abs(torch.div(predictions.float()-(tgt_y+sys.float_info.epsilon).float(),(tgt_y+sys.float_info.epsilon).float()))).item())
            count+=1

    # mean from dataset (ignore if just 1 dataset)
    if count>1:
        mean_sqr_diff=np.mean(mean_sqr_diff)

    if grank==0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time()-et} s')
        #print(tgt_y.float().flatten().cpu().detach().numpy().shape())
        #print(predictions.float().flatten().cpu().detach().numpy().shape())
        if not args.testrun or not args.benchrun:
            #print(tgt_y.float().flatten().cpu().detach().numpy()[:20])
            plot_line(tgt_y.float().flatten().cpu().detach().numpy()[:20], predictions.float().flatten().cpu().detach().numpy()[:20], 'test')
        #if not args.testrun or not args.benchrun:
        #    plot_scatter(inputs[0][0].cpu().detach().numpy(), 
        #            predictions[0][0].cpu().detach().numpy(), 'test')

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
    turb_data = datasets.DatasetFolder(args.data_dir+'latvar_train',\
        loader=hdf5_loader, extensions='.hdf5')
    #turb_data2 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width10_12_16_18/Width1200',\
    #    loader=hdf5_loader, extensions='.hdf5')
    #turb_data3 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width10_12_16_18/Width1600',\
    #    loader=hdf5_loader, extensions='.hdf5')
    #turb_data4 = datasets.DatasetFolder(args.data_dir+'trainfolder/Width10_12_16_18/Width1800',\
    #    loader=hdf5_loader, extensions='.hdf5')

    #turb_data = torch.utils.data.ConcatDataset([turb_data1])#, turb_data2, turb_data3, turb_data4])

    #turb_data = datasets.DatasetFolder(args.data_dir+'trainfolder',\
    #    loader=hdf5_loader, extensions='.hdf5')

    test_data = datasets.DatasetFolder(args.data_dir+'latvar_test',\
         loader=hdf5_loader, extensions='.hdf5')

    # restricts data loading to a subset of the dataset exclusive to the current process
    args.shuff = args.shuff and not args.testrun
    train_sampler = DistributedCustomSampler(
        turb_data, num_replicas=gwsize, rank=grank, shuffle = args.shuff)
    test_sampler = DistributedCustomSampler(
        test_data, num_replicas=gwsize, rank=grank, shuffle = args.shuff)

# distribute dataset to workers
    # persistent workers is not possible for nworker=0
    pers_w = True if args.nworker>1 else False

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    train_loader = torch.utils.data.DataLoader(turb_data, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2,
        sampler=test_sampler, num_workers=args.nworker, pin_memory=True,
        persistent_workers=pers_w, drop_last=True, prefetch_factor=args.prefetch, **kwargs )

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n') 

    # create model

    ## Transfomer model parameters
    dim_val = 1024 
    n_heads = 8 
    n_dec_lay = 6
    n_enc_lay = 6 
    inp_size = 512 #144000 # TODO: TBL input
    num_pred_features = inp_size

    forecast_window = 2

    model = transformer_tbl(
        dim_val=dim_val,
        inp_dim=inp_size,  
        n_dec_lay=n_dec_lay,
        n_enc_lay=n_enc_lay,
        n_heads=n_heads,
        num_pred_features=num_pred_features,
        batch_first=False).to(device)

    #model = nn_reg().to(device)
    #dist.barrier()
    #loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
    #CDM_checkpoint = torch.load('checkpoint.pth.tar',map_location=loc)
    #CDM_model = U_Net().to(device)

    #if args.cuda:
    #    dist_CDM = torch.nn.parallel.DistributedDataParallel(CDM_model,\
    #        device_ids=[device], output_device=device)
    #else:
    #    dist_CDM = torch.nn.parallel.DistributedDataParallel(CDM_model)

    #dist_CDM.load_state_dict(CDM_checkpoint['state_dict'])

# distribute model to workers
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model,\
            device_ids=[device], output_device=device)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model)

# optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr)#, weight_decay=args.wdecay)
    #scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.00001)

# resume state 
    start_epoch = 0
    best_acc = np.Inf
    res_name='/p/scratch/raise-ctp2/sarma1/TR_averaged/checkpoints/CDM_TR_cube/'+'checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(res_name, map_location=loc)
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
    avg_mean_sqr_diff = test(distrib_model, loss_function, device, test_loader, grank, gwsize)

# clean-up
    if grank==0:
        print(f'TIMER: final time: {time.time()-st} s')
    dist.destroy_process_group()

if __name__ == "__main__": 
    main()
    sys.exit()

# eof
