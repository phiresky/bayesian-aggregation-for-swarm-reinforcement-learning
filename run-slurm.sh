#!/bin/bash
#SBATCH --partition=gpu_4,gpu_8
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=20
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=94G

echo "$0" "$@"

module load compiler/gnu/10.2
module load mpi/openmpi
eval "$(conda shell.bash hook)"
conda activate onlypybin
env
nvidia-smi
echo q | htop -C  | tail -c +10

set -eu
numsimul="$1"
shift

./run-simul.sh "$numsimul" 1 "$@"
