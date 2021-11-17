#!/usr/bin/python3
# -*- coding: utf-8 -*-
# author: EI
# version: 210906a
# python script to create a test directory

# libs
import os , glob, shutil

# parameters
cper = 10 # move this percentage to target dir 

# dirs
mDir = os.getcwd()         # main dir
sDir = mDir+'/trainfolder' # source dir
tDir = mDir+'/testfolder'  # target dir

# create main folder
os.makedirs(tDir, exist_ok=True)

# get list of files to be moved and create subfolders on the go
copylist = []
tree1 = glob.glob(sDir+'/*')
for i in tree1:
    #W = i.split('/')[-1]
    W = 'Width1000' # do for this width only  
    tree2 = glob.glob(sDir+'/'+W+'/*')
    os.makedirs(tDir+'/'+W, exist_ok=True)
    for j in tree2:
        L = j.split('/')[-1]
        tree3 = glob.glob(sDir+'/'+W+'/'+L+'/*')
        os.makedirs(tDir+'/'+W+'/'+L, exist_ok=True)
        for k in tree3:
            T = k.split('/')[-1]
            tree4 = glob.glob(sDir+'/'+W+'/'+L+'/'+T+'/*')
            os.makedirs(tDir+'/'+W+'/'+L+'/'+T, exist_ok=True)
            for m in tree4:
                A = m.split('/')[-1]
                tree5 = glob.glob(sDir+'/'+W+'/'+L+'/'+T+'/'+A+'/boxes/*')
                os.makedirs(tDir+'/'+W+'/'+L+'/'+T+'/'+A, exist_ok=True)
                os.makedirs(tDir+'/'+W+'/'+L+'/'+T+'/'+A+'/boxes', exist_ok=True)
                c = 0
                for n in tree5:
                    if c%int(100/cper)==0:
                        copylist.append(n)
                    c+=1
    break # limits W

# move
for i in copylist:
    j = i.split('trainfolder')
    d = j[0]+'testfolder'+j[-1]
    print(d)
    #shutil.copy(i,d) # debug / copy only
    shutil.move(i,d)

# create a list for moved files
textfile = open('moved_data.dat', "w")
textfile.write('source: '+sDir+'\n')
textfile.write('target: '+tDir+'\n')
textfile.write('total files: '+str(len(copylist))+'\n')
for element in copylist:
    textfile.write(element+'\n')
textfile.close()

# eof
