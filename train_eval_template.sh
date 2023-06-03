#!/bin/bash

#SBATCH --job=forl_train_eval
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/home/%u/FoRL/log/%j.out    # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/cluster/home/%u/FoRL/log/%j.err     # where to store error messages
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=8G
#SBATCH --gpus=rtx_3090:1
#SBATCH --mail-type=ALL

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Binary or script to execute
# load module
module load gcc/8.2.0 openblas/0.3.20 python/3.11.2 # cuda/11.8.0 cudnn/8.8.1.3

# use the correct python
export PYTHONPATH=$HOME/FoRL_venv/bin/python

mkdir -p $HOME/log
mkdir -p $HOME/results

$PYTHONPATH $HOME/FoRL/multi_train.py

# We could copy more results from here to output or any other permanent directory

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
