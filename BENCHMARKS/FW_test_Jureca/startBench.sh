#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 220204a
# framework benchmarks


# Parameters
# nodes
ns=1,2,4,8,16
# batch size per GPU
bs=64      
# number of epochs
ep=10
# learning rate
lr=0.001
# number of CPU thread per node
nw=0
# prefetch factor per CPU thread
pf=2
# is this test run (max 4 nodes)?
trun=false

# DO NOT MODIFY BOTTOM
# load JUBE if not loaded
c1=$(module tablelist 2>&1 | grep JUBE)
if [ "$c1" = '  ["JUBE"] = "2.4.1",' ] ; then
  ml JUBE
fi

# modify jobsys at the same time
var1='      <parameter name="iterNO" type="int">'${ns}'</parameter>'
var2='      <parameter name="iterBS" type="int">'${bs}'</parameter>'
var3='      <parameter name="iterEP" type="int">'${ep}'</parameter>'
var4='      <parameter name="iterLR" type="float">'${lr}'</parameter>'
var5='      <parameter name="iterNW" type="int">'${nw}'</parameter>'
var6='      <parameter name="iterPF" type="int">'${pf}'</parameter>'
files1="./*jobsys_AT.xml"
for f in $files1
do
  sed -i "16s|.*|$var1|" ${f}
  sed -i "17s|.*|$var2|" ${f}
  sed -i "18s|.*|$var3|" ${f}
  sed -i "19s|.*|$var4|" ${f}
  sed -i "20s|.*|$var5|" ${f}
  sed -i "21s|.*|$var6|" ${f}
done

# change partition if test run
var7='#SBATCH --partition=dc-gpu'
if [ "$trun" = true ] ; then
  var7='#SBATCH --partition=dc-gpu-devel'
fi
files2="./*startscript"
for f in $files2
do
  sed -i "13s|.*|$var7|" ${f}
done

# modify DS config as well
var8='  "train_batch_size": '${bs}','
sed -i "2s|.*|$var8|" DS_config.json 

# clean folder
if [ "$2" = 'clean' ] ; then
   rm -rf JUBE bench_run
fi

# start JUBE runs
if [ "$1" = 'start' ] ; then
   echo 'starting runs:'
   jube run DDP_jobsys_AT.xml
   jube run Hor_jobsys_AT.xml
   jube run HeAT_jobsys_AT.xml
   jube run DS_jobsys_AT.xml
fi

# end JUBE runs and print results
if [ "$1" = 'end' ] ; then
   echo 'printing results:'
   jube analyse bench_run --id 0 
   jube analyse bench_run --id 1
   jube analyse bench_run --id 2 
   jube analyse bench_run --id 3 

   jube result bench_run --id 0 
   jube result bench_run --id 1
   jube result bench_run --id 2 
   jube result bench_run --id 3 
fi

# error-check
if [ "$1" = 'end' ] || [ "$1" = 'start' ] ; then
   echo 'done'
else
   echo 'usage: bash startBench.sh start clean'
   echo 'options 1: start or end'
   echo 'options 2: clean deletes JUBE and bench_run folders'
fi

# eof
