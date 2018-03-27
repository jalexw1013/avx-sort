#!/bin/bash
#-----------------------------------------------------------------
# Runs serial version of merge throught sbatch on TACC
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J mergeByThreads                 # Job name
#SBATCH -o mergeByThreads.%j.out          # Specify stdout output file (%j expands to jobId)
#SBATCH -p skx-normal         # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                   # Total number of tasks
#SBATCH -t 8:00:00              # Run time (hh:mm:ss) - 8 hours
#SBATCH --mail-user=jwatkins45@gatech.edu
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes

# Launch merge of different sizes.
#./merge -A 1 -B 1
#./merge -A 1 -B 1
#./merge -A 1 -B 1
#./merge -A 5 -B 5
#./merge -A 5 -B 5
#./merge -A 5 -B 5
#./merge -A 20 -B 20
#./merge -A 20 -B 20
#./merge -A 30 -B 30
#./merge -A 32 -B 32
#./merge -A 40 -B 40
#./merge -A 50 -B 50
#./merge -A 100 -B 100
#./merge -A 100 -B 100
#./merge -A 128 -B 128
#./merge -A 128 -B 128
#./merge -A 200 -B 200
#./merge -A 500 -B 500
./merge -F
