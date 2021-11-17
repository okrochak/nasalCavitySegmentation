#!/bin/sh
# author: EI
# version: 210723a
# post-process JUBE benchmark

# get sys info
iDir=$PWD
sysN=$(eval "uname -n | cut -f2- -d.")
echo "system:${sysN}"
echo

# set modules
module --force purge
if [ "$sysN" = 'deepv' ] ; then
   module use $OTHERSTAGES
   ml Stages/2020 GCC Python JUBE
   source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate
elif [ "$sysN" = 'juwels' ] ; then
   ml GCC Python JUBE
   source /p/project/prcoe12/RAISE/envAI_juwels/bin/activate
else
   echo
   echo 'unknown system detected'
   echo 'canceling'
   echo
fi
echo "modules loaded"
echo

gid=$1
if [ "$1" = '' ] ; then
   gid="last"
fi

# extract results
jube analyse bench_run --id $gid

# show results
jube result bench_run --id $gid

# get dir and ID
jDir=$(eval 'jube info bench_run --id $gid | grep Directory: | cut -d " " -f4')
jID=$(eval 'jube info bench_run --id $gid | grep id: | cut -d " " -f3 | cut -d ":" -f2')
jName=$(eval 'jube info bench_run/ --id $gid | grep DDP | head -1 | cut -d " " -f6')

# plot results
python ../plotbench.py $jDir/result/result.dat $jID

# rename
mv bench_id${jID}.png bench_id${jID}_${jName}_${sysN}.png

echo 'results are plotted in bench_id'$jID'_'$jName'_'$sysN'.png'
echo
# eof
