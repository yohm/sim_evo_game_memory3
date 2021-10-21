#!/bin/bash -eux

mpiFCCpx -Nclang -I./eigen -I./json/include -Kopenmp -SCALAPACK -SSL2BLAMP -Kfast -DNDEBUG -std=c++14 -o main_evo_MPI main_evo_MPI.cpp Action.cpp DirectedGraph.cpp StrategyM3.cpp Scalapack.cpp
