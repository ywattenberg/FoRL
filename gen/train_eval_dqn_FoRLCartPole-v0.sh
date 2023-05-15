#!/bin/bash

#SBATCH --job=forl_train_eval_dqn_FoRLCartPole-v0
#SBATCH --time=0:20:00
#SBATCH --output=/cluster/home/%u/forl/log/%j.out    # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/cluster/home/%u/forl/log/%j.err     # where to store error messages
#SBATCH --cpus-per-task=1
# # --mem-per-cpu=2G
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

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Binary or script to execute
# load modules
module load python/3.10.2

$HOME/forl/FoRL_venv/bin/python $HOME/forl/FoRL/train_eval_one.py dqn FoRLCartPole-v0

# We could copy more results from here to output or any other permanent directory

echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0