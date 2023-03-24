#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version: 220323a
# notes: create a dummy dataset via HELPER_Sciprts/createDummyTrainset.py 10 50

# libs
import sys, os, time, numpy as np, h5py, torch, torch.distributed as dist
from torchvision import datasets

class custom_batch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        return self

def collate_batch(batch):
    return custom_batch(batch)

# loader for turbulence HDF5 data
def hdf5_loader(path):
    # read dummy hdf5 files
    f = h5py.File(path, 'r')
    a_group_key = list(f.keys())[0]
    data = np.array(list(f[a_group_key]))

    # convert to torch and remove last ten layers assuming free stream
    return torch.tensor(data,dtype=torch.float32)

def main():
    # start the time.time for profiling
    st = time.time()

    # initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend='nccl')

    # get job rank info - rank==0 master gpu
    global lwsize, gwsize, grank, lrank
    lwsize = torch.cuda.device_count() # local world size - per node
    gwsize = dist.get_world_size()     # global world size - per run
    grank = dist.get_rank()            # global rank - assign per run
    lrank = dist.get_rank()%lwsize     # local rank - assign per node

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda')

# load datasets
    data_dir='./trainfolder_test'
    turb_data = datasets.DatasetFolder(data_dir, loader=hdf5_loader, extensions='.hdf5')

# restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        turb_data, num_replicas=gwsize, rank=grank, shuffle = False)

# distribute dataset to workers
    train_loader = torch.utils.data.DataLoader(turb_data, batch_size=1,
        sampler=train_sampler, collate_fn=collate_batch, num_workers=0, pin_memory=True,
        persistent_workers=False, drop_last=True)

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n')

    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        for batch_ndx, (samples) in enumerate(train_loader):
            res = torch.max(samples.inp)
            print(f'e:{epoch} / c:{batch_ndx} / res = {res} / rank = {grank}')

    # clean-up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    sys.exit()

# eof
