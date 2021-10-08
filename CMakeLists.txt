cmake_minimum_required(VERSION 3.7)
project(cpp)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_VERBOSE_MAKEFILE 1)

set(SOURCE_FILES Action.cpp Action.hpp DirectedGraph.hpp DirectedGraph.cpp
        UnionFind.hpp Partition.hpp StrategyM3.hpp StrategyM3.cpp)
find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH} /usr/local/include/eigen3)

add_executable(test_DirectedGraph test_DirectedGraph.cpp DirectedGraph.hpp DirectedGraph.cpp)

add_executable(test_StrategyM3 test_StrategyM3.cpp ${SOURCE_FILES})
