#!/bin/bash
#-----------------------------------------------------------------
# Runs serial version of merge throught sbatch on TACC
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J merge                 # Job name
#SBATCH -o merge.%j.out          # Specify stdout output file (%j expands to jobId)
#SBATCH -p development           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 68                    # Total number of tasks
#SBATCH -t 00:00:30              # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=jwatkins45@gatech.edu
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes

# Launch merge of different sizes.
#./merge -A 10000 -B 10000
./merge -A 10000000 -B 10000000
#./merge -A 1000000000 -B 1000000000
