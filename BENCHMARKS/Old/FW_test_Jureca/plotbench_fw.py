#!/p/project/raise-ctp1/RAISE/envAI_jureca/bin/python3
# -*- coding: utf-8 -*-
# author: EI
# version 220204a
# speed-up, efficiency and accuracy plots
# plot accuracy when available

# parameters
fsize=10    # font size
xx=10       # size of image
yy=6        # size of image
ws=hs=0.3   # space between subimages
ngpu=4      # ngpu per node
xlm=[0,68]

# libs
import os, sys, matplotlib, getopt, itertools, numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# read data
def readRes(fname):
    return np.genfromtxt(fname,delimiter="|",skip_header=3,usecols=range(1,11))

def plotDat(ax,x,y,l,clr,c):
    mk = ['o','^','d','x']
    ax.plot(x,y,ls='dashed',lw=1.0,marker=mk[c],c=clr[c],mfc=clr[c],mec=clr[c],markersize=5,label=l)
    return ax

# main
def main(argv):
    opts, fname = getopt.getopt(argv,'hi:o:t')

    cm_subsection = np.linspace(0, 1, 9)
    clr = [ plt.cm.Set1(x) for x in cm_subsection]

    # read data
    b1 = readRes('bench_run/000000/result/result.dat') # DDP
    b2 = readRes('bench_run/000001/result/result.dat') # Hor
    b3 = readRes('bench_run/000002/result/result.dat') # Heat
    b4 = readRes('bench_run/000003/result/result.dat') # DS

    # gpus per system
    g1 = b1[:,1]*ngpu 
    g2 = b2[:,1]*ngpu 
    g3 = b3[:,1]*ngpu
    g4 = b4[:,1]*ngpu

    lbl = ['PyTorch-DDP','Horovod','HeAT','DeepSpeed']

    # reset image and generate background
    plt.close('all')
    fig = plt.figure( figsize=(xx,yy) )
    gs = gridspec.GridSpec(2,3)
    gs.update(wspace=ws, hspace=hs)

# comp time
    ax = plt.subplot(gs[0])
    n=7
    ax = plotDat(ax,g1,b1[:,n],lbl[0],clr,0)
    ax = plotDat(ax,g2,b2[:,n],lbl[1],clr,1)
    ax = plotDat(ax,g3,b3[:,n],lbl[2],clr,2)
    ax = plotDat(ax,g4,b4[:,n],lbl[3],clr,3)

    plt.xticks(g1,fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel(r'#GPGPU', fontsize=fsize)
    plt.ylabel(r'comp. time /s', fontsize=fsize)
    plt.xlim(xlm)
    #plt.ylim((1e2,1e4))
    ax.set_yscale('log')
    plt.grid()
    plt.title('(a)',x=0.9,y = 0.85,fontsize=fsize)

# speed-up 
    ax = plt.subplot(gs[1])

    # speed-up
    s1 = b1[0,n]/b1[:,n]
    s2 = b2[0,n]/b2[:,n]
    s3 = b3[0,n]/b3[:,n]
    s4 = b4[0,n]/b4[:,n]

    ax = plotDat(ax,g1,s1,lbl[0],clr,0)
    ax = plotDat(ax,g2,s2,lbl[1],clr,1)
    ax = plotDat(ax,g3,s3,lbl[2],clr,2)
    ax = plotDat(ax,g4,s4,lbl[3],clr,3)

    # ideal
    plt.plot(g1,g1/g1[0],ls='dashed',lw=1.5,c='k',label='ideal')

    plt.xticks(g1,fontsize=fsize)
    plt.yticks(g1/g1[0],fontsize=fsize)
    plt.xlabel(r'#GPGPU', fontsize=fsize)
    plt.ylabel(r'speed-up', fontsize=fsize)
    plt.xlim(xlm)
    #plt.ylim((0,17))
    plt.grid()
    plt.title('(b)',x=0.1,y = 0.85,fontsize=fsize)

    leg=ax.legend(bbox_to_anchor=(-1-ws,1.02,2+ws,.102),loc=3,ncol=5,mode="expand",borderaxespad=0.,\
        fontsize=fsize-1)

# efficiency 
    ax = plt.subplot(gs[3])

    # eff
    e1 = np.minimum(s1*b1[0,1]/b1[:,1],1.0) 
    e2 = np.minimum(s2*b2[0,1]/b2[:,1],1.0) 
    e3 = np.minimum(s3*b3[0,1]/b3[:,1],1.0) 
    e4 = np.minimum(s4*b4[0,1]/b4[:,1],1.0) 

    ax = plotDat(ax,g1,e1,lbl[0],clr,0)
    ax = plotDat(ax,g2,e2,lbl[1],clr,1)
    ax = plotDat(ax,g3,e3,lbl[2],clr,2)
    ax = plotDat(ax,g4,e4,lbl[3],clr,3)

    # ideal
    plt.plot(g1,g1*0+1,ls='dashed',lw=1.5,c='k',label='ideal')

    plt.xticks(g1,fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel(r'#GPGPU', fontsize=fsize)
    plt.ylabel(r'efficiency', fontsize=fsize)
    plt.xlim(xlm)
    #plt.ylim((0,17))
    plt.grid()
    plt.title('(d)',x=0.9,y = 0.85,fontsize=fsize)

# error
    ax = plt.subplot(gs[2])
    m=9
    ax = plotDat(ax,g1,b1[:,m],lbl[0],clr,0)
    ax = plotDat(ax,g2,b2[:,m],lbl[1],clr,1)
    ax = plotDat(ax,g3,b3[:,m],lbl[2],clr,2)
    ax = plotDat(ax,g4,b4[:,m],lbl[3],clr,3)

    plt.xticks(g1,fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.xlabel(r'#GPGPU', fontsize=fsize)
    plt.ylabel(r'training error', fontsize=fsize)
    plt.xlim(xlm)
    #plt.ylim((1e-4,1e-2))
    plt.grid()
    ax.set_yscale('log')
    plt.title('(c)',x=0.9,y = 0.85,fontsize=fsize)

# relative performance 
    ax = plt.subplot(gs[4])

    # interp to integers
    x = np.copy(g1) 
    b1_i = np.interp(x,g1,b1[:,n])
    b2_i = np.interp(x,g2,b2[:,n])
    b3_i = np.interp(x,g3,b3[:,n])
    b4_i = np.interp(x,g4,b4[:,n])

    p1 = b3_i/b1_i*100
    p2 = b3_i/b2_i*100
    p3 = b3_i/b3_i*100
    p4 = b3_i/b4_i*100

    width = 0.2
    xl = np.array([1,2,3,4,5])
    ax.bar(xl - 1.5*width, p1,width,color=clr[0],label=lbl[0])
    ax.bar(xl - 0.5*width, p2,width,color=clr[1],label=lbl[1])
    ax.bar(xl + 0.5*width, p3,width,color=clr[2],label=lbl[2])
    ax.bar(xl + 1.5*width, p4,width,color=clr[3],label=lbl[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'rel. speed-up /%', fontsize=fsize)
    ax.set_xlabel(r'#GPGPU', fontsize=fsize)
    ax.set_xticks(xl)
    ax.set_xticklabels(x)
    ax.set_ylim(ymin=90)
    #ax.set_ylim((90,200))
    plt.grid()
    plt.title('(e)',x=0.1,y = 0.85,fontsize=fsize)

# relative accuracy 
    ax = plt.subplot(gs[5])

    # interp to integers
    b1_i = np.interp(x,g1,b1[:,m])
    b2_i = np.interp(x,g2,b2[:,m])
    b3_i = np.interp(x,g3,b3[:,m])
    b4_i = np.interp(x,g4,b4[:,m])

    p1 = np.sqrt(b1_i/b1_i)*100
    p2 = np.sqrt(b1_i/b2_i)*100
    p3 = np.sqrt(b1_i/b3_i)*100
    p4 = np.sqrt(b1_i/b4_i)*100
    q1 = np.sqrt(b2_i/b1_i)*100
    q2 = np.sqrt(b2_i/b2_i)*100
    q3 = np.sqrt(b2_i/b3_i)*100
    q4 = np.sqrt(b2_i/b4_i)*100

    width = 0.2
    xl = np.array([1,2,3,4,5])
    a = [0,3]
    b = [1,2,4]
    ax.bar(xl[a] - 1.5*width, p1[a],width,color=clr[0],label=lbl[0])
    ax.bar(xl[a] - 0.5*width, p2[a],width,color=clr[1],label=lbl[1])
    ax.bar(xl[a] + 0.5*width, p3[a],width,color=clr[2],label=lbl[2])
    ax.bar(xl[a] + 1.5*width, p4[a],width,color=clr[3],label=lbl[3])
    
    ax.bar(xl[b] - 1.5*width, q1[b],width,color=clr[0],label=lbl[0])
    ax.bar(xl[b] - 0.5*width, q2[b],width,color=clr[1],label=lbl[1])
    ax.bar(xl[b] + 0.5*width, q3[b],width,color=clr[2],label=lbl[2])
    ax.bar(xl[b] + 1.5*width, q4[b],width,color=clr[3],label=lbl[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'rel. $\sqrt{accuracy}$ /%', fontsize=fsize)
    ax.set_xlabel(r'#GPGPU', fontsize=fsize)
    ax.set_xticks(xl)
    ax.set_xticklabels(x)
    #ax.set_ylim(ymin=90)
    #ax.set_ylim((90,400))
    plt.grid()
    plt.title('(f)',x=0.9,y = 0.85,fontsize=fsize)

# save as png
    plt.savefig('bench_fw_jureca.png',dpi=300,bbox_inches='tight',pad_inches=0.2)

if __name__ == "__main__":
    main(sys.argv[1:])

#eof
