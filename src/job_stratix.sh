#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
cd /home/u76572/Pipelined-FFT-OneAPI/src/
make s10

# devcloud_login -b CO walltime=24:00:00 ./job_stratix.sh
# devcloud_login -b S10OAPI walltime=24:00:00 ./job_stratix.sh