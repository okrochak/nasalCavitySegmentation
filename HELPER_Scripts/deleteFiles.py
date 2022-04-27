# -*- coding: utf-8 -*-
# author: EI
# version: 220527a
# python script to delete a lot of folders using many procs
# usage: ./deleteFiles.py <folder>

# libs (some of these needs to be installed via pip)
import sys, os, tqdm, shutil, getopt, multiprocessing, time

# check dataset here
def remove_file(fn,c):
    res = False
    try:
        os.remove(fn)
        res = True
    except:
        pass
    return res, c

def main(argv):
    opts, fname = getopt.getopt(argv,'hi:o:t')
    print(f'removing everying in {fname[0]} using {multiprocessing.cpu_count()} threads\n')
    st = time.time()

    # source dir
    sDir = os.getcwd()+'/'+fname[0]

    # get list of files to be deleted
    slist = []
    for root, dirs, files in os.walk(sDir):
        for file in files:
            slist.append(os.path.join(root,file))

    # create some threads
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # build task list
    tasks = []
    for i in range(len(slist)):
        tasks.append( (slist[i], i ) )

    # assign threads in async manner
    res = [pool.apply_async(remove_file,t) for t in tasks]

    # looking nice counter
    for i in tqdm.tqdm(range(len(tasks))):
        if res[i].get()[0]:
            pass

    # close the threads
    pool.close()
    pool.join()

    # finally remove the empty folder
    shutil.rmtree(sDir)

    # timer
    print(f'finished in {time.time()-st} seconds!')

if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit()

# eof
