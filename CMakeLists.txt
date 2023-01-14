cmake_minimum_required(VERSION 3.7)
project(cpp)


set(CMAKE_CXX_STANDARD 14)

include(CMakeLists.local OPTIONAL)

#set(CMAKE_VERBOSE_MAKEFILE 1)

set(SOURCE_FILES Action.cpp Action.hpp DirectedGraph.hpp DirectedGraph.cpp
        UnionFind.hpp Partition.hpp StrategyM3.hpp StrategyM3.cpp StrategySpace.hpp)
find_package(MPI)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
# See https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html

# change these paths according to your path for homebrew
# paths must be specified to use lapacke
include_directories(SYSTEM /opt/homebrew/opt/lapack/include)
link_directories(/opt/homebrew/opt/lapack/lib)

# unit test for DirectedGraph class
add_executable(test_DirectedGraph test_DirectedGraph.cpp DirectedGraph.hpp DirectedGraph.cpp)

# execute without arguments to run all unit tests for DirectedGraph
#    ./test_StrategyM3
# execute with a single argument to print the details of the strategy
#   ./test_StrategyM3 ALLC
# execute with two arguments to print the game played by these two strategies
#   ./test_StrategyM3 ALLC ALLD
add_executable(test_StrategyM3 test_StrategyM3.cpp ${SOURCE_FILES})
target_link_libraries(test_StrategyM3 PRIVATE Eigen3::Eigen lapacke)

# execute without arguments to run all unit tests for StrategySpace class
add_executable(test_StrategySpace test_StrategySpace.cpp StrategySpace.hpp)
target_link_libraries(test_StrategySpace PRIVATE Eigen3::Eigen lapacke)

# execute without arguments to run all unit tests for GroupedEvoGame class
#   ./test_GroupedEvoGame
# execute with a single argument to print the fixation probabilities of the strategy
#   ./test_GroupedEvoGame ALLC
add_executable(test_GroupedEvoGame test_GroupedEvoGame.cpp GroupedEvoGame.hpp ${SOURCE_FILES})
target_link_libraries(test_GroupedEvoGame PRIVATE Eigen3::Eigen lapacke nlohmann_json::nlohmann_json)

# simulation code for the evolutionary game in grouped-population
add_executable(main_multi_evo main_multi_evo.cpp ${SOURCE_FILES} GroupedEvoGame.hpp)
target_link_libraries(main_multi_evo PRIVATE Eigen3::Eigen lapacke nlohmann_json::nlohmann_json)

# run evolutionary simulations in grouped-population with MPI
add_executable(main_multi_evo_batch main_multi_evo_batch.cpp ${SOURCE_FILES} GroupedEvoGame.hpp)
target_include_directories(main_multi_evo_batch PRIVATE ${PROJECT_SOURCE_DIR}/caravan-lib)
target_link_libraries(main_multi_evo_batch PRIVATE Eigen3::Eigen lapacke MPI::MPI_CXX nlohmann_json::nlohmann_json)

# calculate fractions of efficient, rival, FR strategies in each memory space
add_executable(main_calc_fractions main_calc_fractions.cpp ${SOURCE_FILES})
target_link_libraries(main_calc_fractions PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

# calculate the lifetime starting from a given strategy
add_executable(main_lifetime_distribution main_lifetime_distribution.cpp ${SOURCE_FILES})
target_link_libraries(main_lifetime_distribution PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

# unit test for GroupedEvoGameLowMutation class
add_executable(test_GroupedEvoGameLowMutation test_GroupedEvoGameLowMutation.cpp ${SOURCE_FILES})
target_link_libraries(test_GroupedEvoGameLowMutation PRIVATE Eigen3::Eigen lapacke nlohmann_json::nlohmann_json)

# run evolutionary game in grouped-population with complete-time separation
add_executable(main_multi_evo_low_mut main_multi_evo_low_mut.cpp ${SOURCE_FILES})
target_link_libraries(main_multi_evo_low_mut PRIVATE Eigen3::Eigen lapacke nlohmann_json::nlohmann_json)

# find strategies that have memory lengths shorter than (3,3) from a list of strategies
add_executable(find_shorter_memory_FR find_shorter_memory_FR.cpp)
