#!/bin/bash

# Is specified via environment variable, execute commands in a job scheduled via
# sbatch
if [ "$RUN_ON_NOCTUA" = 1 ]; then
  # Specify a name for the SLURM job
  JOB_NAME="-J ${JOB_NAME:=run}"
  # Hard time limit of the job, will forcefully be terminated if not done within
  # this time
  TIME_LIMIT="-t ${TIME_LIMIT:=01:00:00}"
  # Number of CPUs to use per task
  NUM_CPUS="--cpus-per-task=${NUM_CPUS:=16}"
  # Amount of memory to allocate for the job
  MEM="--mem ${MEM:=64G}"
  # The partition to which the job is submitted
  PARTITION="-p ${PARTITION:=normal}"
  # Notify by mail on all events (queue, start, stop, fail, ...)
  MAIL="--mail-type FAIL --mail-user christoph.berganski@uni-paderborn.de"
  # If using GPUS, specify which type of GPU and how many
  #   Note: Hardcode this to 1, there is no need for more GPUs right now
  if [[ "$PARTITION" = "-p gpu" ]] || [[ "$PARTITION" = "-p dgx" ]]; then
    #   Note: Hardcode this to 1, there is no need for more GPUs right now
    GPUS="--gres=gpu:a100:1"
  fi;
  # Group all sbatch command line arguments into one string
  ARGS="$JOB_NAME $TIME_LIMIT $NUM_CPUS $MEM $PARTITION $MAIL $GPUS"
  # Forward all arguments following the shell script to be executed as the
  # command line inside of another shell script which is executed via sbatch
  #   Note: Waiting/Blocking sbatch with the -W option
  # shellcheck disable=SC2086
  sbatch $ARGS -W --verbose noctua.sh "$@"
# By default, execute the job locally
else
  # Forward all arguments following the shell script to be executed as the
  # command line within this script
  eval "$@";
fi;
