#!/bin/bash

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
export SINGULARITY_CACHEDIR=/dev/shm/singularity-cache/
export SINGULARITY_TMPDIR=/dev/shm/singularity-tmp/
export FINN_SINGULARITY=$PC2DATA/hpc-prf-ekiapp/FINN_IMAGES/xilinx/finn_dev.sif

# Prepare FINN to find the Vitis/Vivado installation
export FINN_XILINX_PATH=/opt/software/FPGA/Xilinx/
export FINN_XILINX_VERSION=2022.2

# Somehow these options are required to get FINN running on the cluster...
export LC_ALL="C"
export PYTHONUNBUFFERED=1
export XILINX_LOCAL_USER_DATA="no"

# If a path to a FINN installation is specified, move it to some faster storage
# location
if [[ -d "$FINN" ]]; then
  # Copy FINN to the ramdisk
  cp -r "$FINN" /dev/shm/finn/
  # Redirect the path specified via environment variable to use the copy
  export FINN=/dev/shm/finn/
fi;

# Generate FINN build outputs and temporaries to the ramdisk
export FINN_HOST_BUILD_DIR=/dev/shm/finn-build

# Write the command line to be executed to the log
echo "$@"
# Forward all command line arguments as the command line to be run as the job
eval "$@"

# If FINN actually produced build outputs
if [[ -d "$FINN_HOST_BUILD_DIR" && $DEBUG ]]; then
  # Generate a (hopefully) unique name for debugging output
  DEBUG_OUTPUT="build-$(hostname)-$(date +'%Y-%m-%d-%H-%M-%S').tar.gz"
  # For debugging purposes collect all build outputs from the ramdisk
  tar -zcf "$DEBUG_OUTPUT" /dev/shm/finn-build
fi;
