# -*- coding: utf-8 -*-
# author: EI
# version: 210927a
# python script to verify each dataset on given folder
# usage: ./verifyDataSet.py <folder>

# libs
import sys, os, glob, h5py, tqdm, getopt, multiprocessing

# check dataset here
def check_ds(fn,c):
    res = True
    try:
        # open file
        test = h5py.File(fn,'r')
        # close file
        test.close()
        # result
        res = False
    except:
        pass
    return res,c

def main(argv):
    opts, fname = getopt.getopt(argv,'hi:o:t')
    print(f'checking {fname[0]} for broken datasets\n')

    # source dir
    sDir = os.getcwd()+'/'+fname[0]
    
    # get list of files to be verified 
    slist = glob.glob(sDir+'/**/*hdf5', recursive=True)
    
    # create some threads
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # build task list
    tasks = []
    for i in range(len(slist)):
        tasks.append( (slist[i], i ) )

    # assign threads in async manner
    res = [pool.apply_async(check_ds,t) for t in tasks]

    # create list of broken datasets
    testlist = []
    for i in tqdm.tqdm(range(len(res))):
        if res[i].get()[0]:
            testlist.append(slist[i])

    # close the threads
    pool.close()
    pool.join()
    
    # output the list for broken files
    if len(testlist)>0:
        textfile = open('broken_data.dat', "w")
        textfile.write('source: '+sDir+'\n')
        textfile.write('total files: '+str(len(testlist))+'\n')
        for element in testlist:
            textfile.write(element+'\n')
        textfile.close()
        
        print(f'broken datasets are written to broken_data.dat!')
    else:
        print(f'all datasets are OK!')

if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit()

# eof
