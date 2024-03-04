#!/bin/bash

# Is specified via environment variable, execute commands in a job scheduled via
# sbatch
if [ "$RUN_SBATCH" = 1 ]; then
  # Forward all arguments following the shell script to be executed as the
  # command line inside of another shell script which is executed via sbatch
  #   Note: Waiting/Blocking sbatch with the -W option
  sbatch -W batch.sh "$@"
# By default, execute the job locally
else
  # Forward all arguments following the shell script to be executed as the
  # command line within this script
  "$@";
fi;
