#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
cd /home/u76572/Pipelined-FFT-OneAPI/src/
make s10

# devcloud_login -b CO walltime=24:00:00 ./job_stratix.sh
# devcloud_login -b S10OAPI walltime=24:00:00 ./job_stratix.sh
# check running tasks: watch -n 1 qstat -n -1
# delete all tasks: qselect -u u76572 | xargs qdel