#!/bin/bash

# Give a name to the job which can be configured from the outside
#SBATCH -J "${JOB_NAME:=run}"
# Hard time limit of the job, will forcefully be terminated if not done within
# this time
#SBATCH -t "${TIME_LIMIT:=01:00:00}"

# Number of CPUs ti use per task
#SBATCH --cpus-per-task="${NUM_CPUS:=16}"
# AMount of memory to allocate for the job
#SBATCH --mem "${MEM:=64G}"
# The partition to which the job is submitted
#SBATCH -p "${PARTITION:=normal}"

# Notify by mail on all events (queue, start, stop, fail, ...)
#SBATCH --mail-type ALL
#SBATCH --mail-user christoph.berganski@uni-paderborn.de

# If using GPUS, specify which type of GPU and how many
#   Note: Hardcode this to 1, there is no need for more GPUs right now
#SBATCH --gres=gpu:a100:1

# Setup the python environment for model training and evaluation and export
module load lang/Python/3.10.4-GCCcore-11.3.0
python3.10 -m venv /dev/shm/env/
source /dev/shm/env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup the FPGA development environment
module load fpga
module load xilinx/xrt/2.14
module load xilinx/vitis/22.2

# Prepare for running FINN in Singularity containers
module load system singularity
export SINGULARITY_CACHEDIR=/dev/shm/
export SINGULARITY_TMPDIR=/dev/shm/
export FINN_SINGULARITY=$PC2DATA/hpc-prf-ekiapp/FINN_IMAGES/xilinx/finn_dev.sif

# Prepare FINN to find the Vitis/Vivado installation
export FINN_XILINX_PATH=/opt/software/FPGA/Xilinx/
export FINN_XILINX_VERSION=2022.2

# Forward all command line arguments as the command line to be run as the job
"$@"
