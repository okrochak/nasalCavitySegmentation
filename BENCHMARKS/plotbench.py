#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: EI
# version 210729a
# speed-up, efficiency and accuracy plots
# plot accuracy when available

# parameters
fsize=10   # font size
xx=yy=7    # size of image
ws=hs=0.3  # space between subimages

# libs
import os, sys, matplotlib, getopt, itertools, numpy as np, pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# read data
def readRes(fname):
    try:
        ben1 = np.genfromtxt(fname[0],delimiter="|",skip_header=3,usecols=range(1,9))
    except:
        ben1 = np.genfromtxt(fname[0],delimiter="|",skip_header=3,usecols=range(1,8))
    return ben1

# get accuracy from multi gpus
def getRes(fname,no):
    mainDir = os.getcwd()
    ben2 = np.zeros((len(no),4 ))
    for i in range(len(no)):
        # check each job.out file for avg_mean_sqr_diff
        os.chdir('./JUBE/'+fname[1]+'_'+str(int(i)))
        file = open('job.out').readlines()
        extract=np.zeros((1))
        for lines in file:
            if 'avg_mean_sqr_diff' in lines:
                extract = np.append(extract,float(lines.split(':')[2].strip().split(' ')[0]))
        ext = np.delete(extract,0)
        os.chdir(mainDir)

        # create matrix of results
        ben2[i,0] = np.amin(ext)
        ben2[i,1] = np.amax(ext)
        ben2[i,2] = np.mean(ext)
        ben2[i,3] = np.std(ext)
    return ben2

# main
def main(argv):
    opts, fname = getopt.getopt(argv,'hi:o:t')

    # initialise
    cm_subsection = np.linspace(0, 1, 9)
    clr = [ plt.cm.Set1(x) for x in cm_subsection]
    mk = ['o','^','d','x']

    # read data
    ben1 = readRes(fname)

    # copy to vars
    no=ben1[:,0] # number
    nn=ben1[:,1] # nodes
    bs=ben1[:,2] # batch size
    ep=ben1[:,3] # #epochs
    ds=ben1[:,4] # data size or learning rate
    tt=ben1[:,5] # comp time
    mm=ben1[:,6] # memory
    try:
        ac=ben1[:,7] # accuracy
    except:
        pass

    # if variable learning rate, fix it to the initial value
    if ds[1]!=ds[0]:
        ds = ds*0+ds[0]

    # get training errors
    try:
        ben2 = getRes(fname,no)
        ac_min=ben2[:,0] # min err
        ac_max=ben2[:,1] # max err
        ac_mea=ben2[:,2] # mean err
        ac_std=ben2[:,3] # std err
    except:
        pass

    # get unique sets
    bs_u = np.unique(bs)
    ep_u = np.unique(ep)
    ds_u = np.unique(ds)

    # make x axis
    xaxis = np.arange(0,len(nn[bs==bs_u[0]]))

    # reset image and generate background
    plt.close('all')
    fig = plt.figure( figsize=(xx,yy) )
    gs = gridspec.GridSpec(2,2)
    gs.update(wspace=ws, hspace=hs)

# comp time
    ax = plt.subplot(gs[0])
    c=0
    for i,j,k in itertools.product(bs_u,ep_u,ds_u):
        mask = ((bs==i) & (ep==j) & (ds==k))
        nnC = nn[mask]
        ttC = tt[mask]
        plt.plot(xaxis,ttC,ls='dashed',lw=1.0,marker=mk[c],c=clr[c],mfc=clr[c],mec=clr[c],markersize=5)
        c+=1

    plt.xticks(xaxis,fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel(r'#nodes', fontsize=fsize)
    plt.ylabel(r'comp. time /s', fontsize=fsize)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in xaxis:
        if i%2==0:
            labels[i] = int(nnC[i])
        else:
            labels[i] = ''
    ax.set_xticklabels(labels)

# speed-up
    ax = plt.subplot(gs[1])
    c=0
    for i,j,k in itertools.product(bs_u,ep_u,ds_u):
        mask = ((bs==i) & (ep==j) & (ds==k))
        nnC = nn[mask]
        ttC = tt[mask]
        suC = ttC[0]/ttC
        plt.plot(xaxis,suC,ls='dashed',lw=1.0,marker=mk[c],c=clr[c],mfc=clr[c],mec=clr[c],markersize=5,\
            label='BS:'+str(int(i))+'/EP:'+str(int(j))+'/LR:'+str(float(k)))
        c+=1

    # ideal
    plt.plot(xaxis,nnC/nnC[0],ls='dashed',lw=1.5,c='k',label='ideal')

    plt.xticks(xaxis,fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel(r'#nodes', fontsize=fsize)
    plt.ylabel(r'speed up', fontsize=fsize)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in xaxis:
        if i%2==0:
            labels[i] = int(nnC[i])
        else:
            labels[i] = ''
    ax.set_xticklabels(labels)

    leg=ax.legend(bbox_to_anchor=(-1-ws,1.02,2+1*ws,.102),loc=3,ncol=4,mode="expand",borderaxespad=0.,\
        fontsize=fsize-1)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

# efficiency
    ax = plt.subplot(gs[2])
    c=0
    for i,j,k in itertools.product(bs_u,ep_u,ds_u):
        mask = ((bs==i) & (ep==j) & (ds==k))
        nnC = nn[mask]
        ttC = tt[mask]
        efC = np.minimum(ttC[0]/ttC*nnC[0]/nnC,1.0)
        plt.plot(xaxis,efC,ls='dashed',lw=1.0,marker=mk[c],c=clr[c],mfc=clr[c],mec=clr[c],markersize=5)
        c+=1

    # ideal
    plt.plot(xaxis,nnC*0+1,ls='dashed',lw=1.5,c='k',label='ideal')

    plt.xticks(xaxis,fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel(r'#nodes', fontsize=fsize)
    plt.ylabel(r'efficiency', fontsize=fsize)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in xaxis:
        if i%2==0:
            labels[i] = int(nnC[i])
        else:
            labels[i] = ''
    ax.set_xticklabels(labels)

# accuracy
    try:
        ax = plt.subplot(gs[3])
        c=0
        for i,j,k in itertools.product(bs_u,ep_u,ds_u):
            mask = ((bs==i) & (ep==j) & (ds==k))
            nnC = nn[mask]
            acC = ac[mask]
            # error-bar
            try:
                ac1C = ac_min[mask]
                ac2C = ac_max[mask]
                ac3C = ac_mea[mask]
                ac4C = ac_std[mask]
                plt.errorbar(xaxis,ac3C,yerr=ac4C,capsize=3,\
                    ls='dashed',lw=1.0,marker=mk[c],c=clr[c],mfc=clr[c],mec=clr[c],markersize=5)
            # accuracy - if available
            except:
                plt.plot(xaxis,acC,ls='dashed',lw=1.0,marker=mk[c],c=clr[c],mfc=clr[c],mec=clr[c],markersize=5)
            c+=1

        # ideal
        lbl = r'training error'
        if np.amax(acC)>50:
            plt.plot(xaxis,nnC*0+100,ls='dashed',lw=1.5,c='k',label='ideal')
            lbl = r'accuracy /%'
        else:
            ax.set_yscale('log')

        plt.xticks(xaxis,fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.xlabel(r'#nodes', fontsize=fsize)
        plt.ylabel(lbl, fontsize=fsize)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i in xaxis:
            if i%2==0:
                labels[i] = int(nnC[i])
            else:
                labels[i] = ''
        ax.set_xticklabels(labels)
    except:
        print('accuracy data not found!')
        pass

# save as png
    plt.savefig('bench_id'+str(fname[1])+'.png',dpi=300,bbox_inches='tight',pad_inches=0.2)

if __name__ == "__main__":
    main(sys.argv[1:])

#eof
