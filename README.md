# Evolutionary game for memory-3 strategy space in well-mixed or group-structured populations

This is the source code repository for the paper [TBA].

## Prerequisites

- cmake
- lapack
- Eigen
- OpenMP
- MPI

On macOS, install the libraries using homebrew as follows.

```shell
brew install cmake
brew install lapack
brew install eigen3
brew install nlohmann-json
brew install libomp
brew install openmpi
```

## Build

```shell
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make
```

You can find the usage of each executable in the `CMakelists.txt` file.

## Execution

Run the shell script in `script_low_mut` or `script_multievo` for the simulations in well-mixed and group-structured populations, respectively.
Before running the script, follow the instruction in `readme.md` to install python packages.

After these python packages are installed, prepare `_input.json` file. The file contains input parameters.
You can find sample json files in each directory.

With `_input.json` in the current directory, run the script as follows:

```shell
./run.sh
```

## On Fugaku

A build script for Fugaku is the following. Change the path to the Eigen library appropriately.

```shell
#!/bin/bash -eux

mpiFCCpx -Nclang -Ijson/include -I$HOME/data/sandbox/eigen-3.3.7 -Kfast -Kopenmp -o main_multi_evo_batch main_multi_evo_batch.cpp StrategyM3.cpp Action.cpp DirectedGraph.cpp -SSL2
```

Change the path of Eigen accordingly.

A job submission sample script for Fugaku:

```shell
#!/bin/sh
#PJM --rsc-list "node=400"
#PJM --rsc-list "elapse=24:00:00"
#PJM --rsc-list "rscgrp=large"
#PJM --mpi "max-proc-per-node=48"
#PJM -S

mpiexec -stdout-proc ./%/1000R/stdout -stderr-proc ./%/1000R/stderr ../main_multi_evo_batch _input_batch.json
```

A sample of `_input_batch.json` is as follows:

```json
{
  "T_max": 1000000,
  "T_print":1000,
  "T_init": 100000,
  "M": 1000,
  "N": 2,
  "benefit": [1.5,3.0,6.0],
  "error_rate": 0.000001,
  "sigma_in_b": 30.0,
  "sigma_out_b": [3.0, 30.0],
  "p_nu": [0.1, 0.04,0.02,0.01, 0.004,0.002,0.001, 0.0004,0.0002,0.0001, 0.00004,0.00002,0.00001],
  "strategy_space": [[3,3],[1,1]],
  "weighted_sampling": 1,
  "initial_condition": "random",
  "excluding_strategies": [],
  "number_of_runs": 100,
  "_seed": 1234567890
}
```
