#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# author: EI
# version: 220323a
# python script to create a dummy training dataset 

# libs
import os, sys, getopt, h5py, numpy as np

# create dirs
os.makedirs('trainfolder_test', exist_ok=True)
os.makedirs('trainfolder_test/Width', exist_ok=True)

def main(argv):
    try:
        opts, fname = getopt.getopt(argv,'hi:o:t')
        n1,n2 = int(fname[0]),int(fname[1])
    except:
        print('usage: python createDummyTrainset.py n1 n2')
        print('n1: dataset size')
        print('n2: #dataset')
        sys.exit()

    # create ones
    res = np.ones((n1,n1,n1))

    # create many datasets in h5 format
    for c in range(n2):
        # each dataset has unique values
        out = res * c

        # create h5s
        h5f = h5py.File('trainfolder_test/Width/test_'+str(c)+'.hdf5', 'w')
        h5f.create_dataset('out', data=out)
        h5f.close()

if __name__ == "__main__":
    main(sys.argv[1:])
    print('done!')

# eof
