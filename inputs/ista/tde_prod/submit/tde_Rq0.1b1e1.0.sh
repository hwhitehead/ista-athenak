#!/bin/bash
##----------------------------------
# single CPU example
#----------------------------------

#SBATCH --job-name=tde_hydro
#SBATCH -o /nfs/scistore24/haimagrp/hwhitehe/athenak_wdir/tde_prod/logs/%x_%j.out
#SBATCH -e /nfs/scistore24/haimagrp/hwhitehe/athenak_wdir/tde_prod/logs/%x_%j.err

# number of CPUs to be used
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

#SBATCH --partition=gpu100
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
# SBATCH --constraint=2080ti
#SBATCH --constraint=H100

# Define the number of hours the job should run.
# Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=2:00:00

# Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem-per-cpu=8G

# Pick whether you prefer requeue or not. If you use the --requeue
# option, the requeued job script will start from the beginning,
# potentially overwriting your previous progress, so be careful.
# For some people the --requeue option might be desired if their
# application will continue from the last state.
# Do not requeue the job in the case it fails.
#SBATCH --no-requeue

# Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# for single-CPU jobs make sure that they use a single thread
export OMP_NUM_THREADS=1

# load relevant software modules
module load python/3.13
module load cuda

# activate venv
# source /nfs/scistore24/haimagrp/hwhitehe/cuDART/pysrc/cudart_venv/bin/activate

# move to athenak directory
export ATHENAK_DIR="/nfs/scistore24/haimagrp/hwhitehe/ista-athenak"
export ATHENAK_WDIR="/nfs/scistore24/haimagrp/hwhitehe/athenak_wdir"
cd $ATHENAK_DIR

# call executable
srun --cpu_bind=verbose ./build/src/athena -i inputs/ista/tde_prod/tde_Rq0.1b1e1.0.athinput -d ${ATHENAK_WDIR}/tde_prod/tde_Rq0.1b1e1.0

