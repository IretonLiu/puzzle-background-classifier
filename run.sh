#!/bin/bash
# partition name
#SBATCH -p batch
# number of nodes
#SBATCH -N 1
# or
##SBATCH --nodes=2
# number of cores
##SBATCH -c 12
# size of memory pool
##SBATCH --mem 10
# time limit for the job
##SBATCH -t 5:00
# job name
#SBATCH -J gmm
# name of output file
#SBATCH -o /home-mscluster/iliu/CV/Assignment/output/%N.%j.out
# name of error file
#SBATCH -e /home-mscluster/iliu/CV/Assignment/error/%N.%j.out

python3 src/main.py
