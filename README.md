# Evolutionary game for memory-3 strategy space in well-mixed or group-structured populations

This is the source code repository for the paper [TBA].

## Prerequisites

- cmake
- lapack
- Eigen
- OpenMP
- MPI

On macOS,

```shell
brew install cmake
brew install lapack
brew install eigen3
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

## Execution

Run the shell script in `script_low_mut` or `script_multievo` for the simulations in well-mixed and group-structured populations, respectively.
Before running the script, follow the instruction in `readme.md` to install python packages.

After these python packages are installed, prepare `_input.json` file. The file contains input parameters.
You can find sample json files in each directory.

With `_input.json` in the current directory, run the script as follows:

```shell
./run.sh
```

