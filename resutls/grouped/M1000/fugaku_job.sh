#!/bin/sh
#PJM --rsc-list "node=400"
#PJM --rsc-list "elapse=24:00:00"
#PJM --rsc-list "rscgrp=large"
#PJM --mpi "max-proc-per-node=48"
#PJM -S

mpiexec -stdout-proc ./%/1000R/stdout -stderr-proc ./%/1000R/stderr ../main_multi_evo_batch _input_batch.json

