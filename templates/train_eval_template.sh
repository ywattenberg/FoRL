#!/bin/bash

#SBATCH --job=forl_train_eval_<ALGO>_<ENV>
#SBATCH --time=0:20:00
#SBATCH --output=/cluster/home/%u/forl/log/%j.out    # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/cluster/home/%u/forl/log/%j.err     # where to store error messages
# # --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=8G


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
# load modules
module load gcc/8.2.0
module load python/3.10.4

# use the correct python
export PYTHONPATH=$HOME/forl/FoRL_venv/bin/python

mkdir -p $HOME/forl/log
mkdir -p $HOME/forl/results

$PYTHONPATH $HOME/forl/FoRL/train_eval_one.py <ALGO> <ENV>

# We could copy more results from here to output or any other permanent directory

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0