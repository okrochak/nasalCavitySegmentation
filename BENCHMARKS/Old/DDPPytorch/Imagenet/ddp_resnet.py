import argparse, sys, platform, os, time, numpy as np
from timeit import default_timer as timer

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali import pipeline_def, Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator

# lars external implementation by https://github.com/binmakeswell/LARS-ImageNet-PyTorch
from lars import *




# training settings
def parsIni():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--backend', type=str, default='nccl', metavar='N',
                        help='backend for parrallelisation (default: nccl)')
    return parser


def test(model, device, test_loader, lrank, grank, world_size):
    model.eval()
    test_loss = 0
    #with model.no_sync():
    with torch.no_grad():
        for (data,) in test_loader:
            data, target = data['data'], data['label'].squeeze(-1).long()
            target = target - 1 
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            test_acc = accuracy(output,target)
            output,target = output.to(device), target.to(device)
    test_loss /= len(test_loader)
    test_acc.to(device)
    test_acc.contiguous()
    test_acc = reduce_tensor(test_acc, world_size)
    return test_acc.item()

def train(args, model, device, train_loader, optimizer, epoch, grank, bs):
    
    model.train()
    train_acc = 0
    
    for batch_idx, (data, ) in enumerate(train_loader):
        
        #if grank==0:
        #    print(batch_idx)       
        
        #adjust_learning_rate(args.lr, optimizer, epoch, batch_idx, len(train_loader), bs)
        data, target = data['data'], data['label'].squeeze(-1).long()
        target = target - 1 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()       
        output = model(data)
        train_acc = accuracy(output,target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
    #lr_scheduler.step()
        #annealer.step()

    #torch.distributed.all_reduce(train_acc)
    if (torch.distributed.get_rank()==0):
        print("Local Train Loss: {} Train acc: {}".format(loss, train_acc))        
        #print("LR: ", param_group['lr'])
        

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    #return pred.eq(target.view_as(pred)).cpu().float().mean()
    return pred.eq(target.view_as(pred)).float().mean()
        

def adjust_learning_rate(lr, optimizer, epoch, step, len_epoch, bs):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    
    #lr = lr*(int(os.environ.get("SLURM_NNODES", 0))*torch.distributed.get_world_size())*bs/256
    lr = lr*torch.distributed.get_world_size()
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr       
        
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt
#
# MAIN
#
#
def main():
    # get parse args
    parser = parsIni()
    args = parser.parse_args()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    
    #grank = int(os.environ.get("SLURM_PROCID", 0))
    #lrank = int(os.environ.get("SLURM_LOCALID", 0))
    #print("grank: {} lrank: {}".format(grank,lrank))
    #print(torch.cuda.current_device())
    
    #print("NTASKS", int(os.environ['SLURM_NTASKS']))
    #print("PROCID", int(os.environ['SLURM_PROCID']))
    
    # initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #updated code
    st = timer()
    torch.distributed.init_process_group(backend=args.backend)
    grank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    lwsize = torch.cuda.device_count()
    lrank = torch.distributed.get_rank()%lwsize
    
    if (grank==0):
        print("World size: ", world_size)
    print("GLobal rank: ", grank)
    torch.distributed.barrier()

    # some debug
    if grank==0: 
        print('TIMER: initialise:', timer()-st, 's') 
        print('DEBUG: sys.version:',sys.version)
        print('DEBUG: args.data_dir:',args.data_dir)
        print('DEBUG: args.batch_size:',args.batch_size)
        print('DEBUG: args.epochs:',args.epochs)
        print('DEBUG: args.backend:',args.backend,'\n')

    torch.cuda.set_device(lrank)
    device = torch.cuda.current_device()

    # Have all workers wait - prints errors at the moment
    #torch.distributed.barrier()

    
    # DALI DATALOADER
    train_tfrecord = sorted(glob.glob('/p/scratch/cslfse/aach1/datasets/imagenet-1K-tfrecords/train-*-of-01024'))
    train_tfrecord_idx =sorted(glob.glob('/p/scratch/cslfse/aach1/datasets/imagenet-1K-tfrecords/idx_files/train-*-of-01024.idx'))

    
    val_tfrecord = sorted(glob.glob('/p/scratch/cslfse/aach1/datasets/imagenet-1K-tfrecords/validation-*-of-00128'))
    val_tfrecord_idx =sorted(glob.glob('/p/scratch/cslfse/aach1/datasets/imagenet-1K-tfrecords/idx_files/validation-*-of-00128.idx'))

    def common_pipeline(jpegs, labels, training):
        
        if training: 
            images = fn.decoders.image(jpegs, device='cpu')

            images = fn.resize(
                images,
                resize_shorter=fn.random.uniform(range=(256, 480)),
                interp_type=types.INTERP_TRIANGULAR)

            images = fn.crop_mirror_normalize(
                images,
                crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
                crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
                dtype=types.FLOAT,
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                mirror=fn.random.coin_flip(probability=0.5))
        else:
            images = fn.decoders.image(jpegs, device='cpu')
            
            images = fn.resize(
                images,
                resize_shorter=(224),
                interp_type=types.INTERP_TRIANGULAR)

            images = fn.crop_mirror_normalize(
                images,
                dtype=types.FLOAT,
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                mirror=False)
                        
        return images, labels


    @pipeline_def
    def tfrecord_reader_pipeline(path, index_path, training, shard_id, num_shards):
        inputs = fn.readers.tfrecord(
            path = path,
            index_path = index_path,
            features = {
                "image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)},
            random_shuffle=True,
            shard_id=shard_id, 
            num_shards=num_shards, 
            name='Reader')


            
        return common_pipeline(inputs["image/encoded"], inputs["image/class/label"], training)

    train_pipe = tfrecord_reader_pipeline(path=train_tfrecord, index_path=train_tfrecord_idx,
        batch_size=args.batch_size, num_threads=1, device_id=torch.cuda.current_device(), training=True, shard_id=grank, num_shards=world_size) 
    
    val_pipe = tfrecord_reader_pipeline(path=val_tfrecord, index_path=val_tfrecord_idx,
        batch_size=args.batch_size, num_threads=4, device_id=torch.cuda.current_device(), training=False, shard_id=grank, num_shards= world_size)
                                       
    train_pipe.build()
    val_pipe.build()
    
    dali_train_iter = DALIGenericIterator(train_pipe, ['data', 'label'], reader_name='Reader', auto_reset=True)              
    dali_val_iter = DALIGenericIterator(val_pipe, ['data', 'label'], reader_name='Reader', auto_reset=True)                  
               
    if grank==0: 
        print('TIMER: build data piplines:', timer()-st, 's') 

    Net = models.resnet50(False)
        
    # create CNN model
    model = Net.to(device)

    # Optimizer step
    #optimizer = torch.optim.Adam(mel.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    #optimizer = create_optimizer_lars(model=model, lr=args.lr, epsilon=1e-5, momentum=0.9, weight_decay=0.00005, bn_bias_separately=True) 
    
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 90, eta_min=0, last_epoch=-1, verbose=False)
    
    
    #lr_scheduler = PolynomialWarmup(optimizer, decay_steps=args.epochs * num_steps_per_epoch,
    #                            warmup_steps=5 * num_steps_per_epoch,
    #                            end_lr=0.0, power=2.0, last_epoch=-1)
    
    
    # distribute
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model,\
            device_ids=[device], output_device=device)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model)

    if grank==0: 
        print('TIMER: broadcast:', timer()-st, 's') 
        print(f'\nDEBUG: start training') 
        print(f'--------------------------------------------------------') 

    et = timer()
    acc_list = []
    for epoch in range(1, args.epochs + 1):
        if (grank==0): 
            print("Epoch: ", epoch)
        lt = timer()
        train(args, distrib_model, device, dali_train_iter, optimizer, epoch, grank, args.batch_size)
        acc_test = test(distrib_model, device, dali_val_iter, lrank, grank, world_size)
        if epoch + 1 == args.epochs:
            dali_train_iter.last_epoch = True
            dali_val_iter.last_epoch = True
        if grank==0: 
            print('TIMER: epoch time:', timer()-lt, 's')
            print('DEBUG: accuracy:', acc_test)
        acc_list.append(acc_test)
        torch.distributed.barrier(async_op=True)

    # finalise
    torch.distributed.barrier()
    if grank==0: 
        print(f'\n--------------------------------------------------------') 
        print('DEBUG: results:\n') 
        print('TIMER: last epoch time:', timer()-lt, 's') 
        print('TIMER: total epoch time:', timer()-et, 's')
        print('DEBUG: last accuracy:', acc_test)
        print("Best test accuracy: ", max(acc_list))


    if grank==0 and args.cuda:
        print('DEBUG: memory req:',int(torch.cuda.memory_reserved(lrank)/1024/1024),'MB')
    elif not args.cuda:
        print('DEBUG: memory req: 0 MB')

if __name__ == "__main__": 
    main()

#eof
