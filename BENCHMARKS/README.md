# benchmarks using JUBE

## load JUBE
`ml JUBE`

## setup benchmarks
everything you need is already in `general_jobsys.xml`
*modify general_jobsys.xml if needed*

## run benchmark
`jube run general_jobsys.xml`

## run benchmark on development partitions
`jube run general_jobsys.xml --tag devel`

## check if finalized
`jube continue bench_run --id last`

## show results
`jube result -a bench_run --id last`\
this will create `result-csv.dat` file in `results` folder

## some more information
1. `jube info bench_run --id last`
2. `jube log bench_run --id last`

## print results
use `plotbench.ipynb` (or `plotbench.py` for paper quality images)\
*note: `result-csv.dat` is now needed here*

## latest results are in `Results` folder
benchmark using Synthetic Data for TBL\
old but good results are still available in `Old` folder

## contact:
EI
