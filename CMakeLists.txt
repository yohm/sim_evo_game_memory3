cmake_minimum_required(VERSION 3.7)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_OSX_DEPLOYMENT_TARGET 11.6)

#set(CMAKE_VERBOSE_MAKEFILE 1)

set(SOURCE_FILES Action.cpp Action.hpp DirectedGraph.hpp DirectedGraph.cpp
        UnionFind.hpp Partition.hpp StrategyM3.hpp StrategyM3.cpp StrategySpace.hpp)
find_package(MPI)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
# See https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html

include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/json/include ${CMAKE_SOURCE_DIR}/eigen /usr/local/opt/lapack/include)
link_directories(/usr/local/opt/lapack/lib)

add_executable(test_DirectedGraph test_DirectedGraph.cpp DirectedGraph.hpp DirectedGraph.cpp)

add_executable(test_StrategyM3 test_StrategyM3.cpp ${SOURCE_FILES})
target_link_libraries(test_StrategyM3 PRIVATE Eigen3::Eigen lapacke)

add_executable(test_StrategySpace test_StrategySpace.cpp StrategySpace.hpp)
target_link_libraries(test_StrategySpace PRIVATE Eigen3::Eigen lapacke)

add_executable(test_GroupedEvoGame test_GroupedEvoGame.cpp GroupedEvoGame.hpp ${SOURCE_FILES})
target_link_libraries(test_GroupedEvoGame PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(main_multi_evo main_multi_evo.cpp ${SOURCE_FILES} GroupedEvoGame.hpp)
target_link_libraries(main_multi_evo PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(main_multi_evo_batch main_multi_evo_batch.cpp ${SOURCE_FILES} GroupedEvoGame.hpp)
target_include_directories(main_multi_evo_batch PRIVATE ${PROJECT_SOURCE_DIR}/caravan-lib)
target_link_libraries(main_multi_evo_batch PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX MPI::MPI_CXX)

add_executable(test_GroupedEvoGameLowMutation test_GroupedEvoGameLowMutation.cpp ${SOURCE_FILES})
target_link_libraries(test_GroupedEvoGameLowMutation PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(main_multi_evo_low_mut main_multi_evo_low_mut.cpp ${SOURCE_FILES})
target_link_libraries(main_multi_evo_low_mut PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(main_mem1_finite_mut main_mem1_finite_mut.cpp ${SOURCE_FILES})
target_link_libraries(main_mem1_finite_mut PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(main_mem1_finite_mut_analysis main_mem1_finite_mut_analysis.cpp ${SOURCE_FILES})
target_link_libraries(main_mem1_finite_mut_analysis PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(main_mem1_low_mut main_mem1_low_mut.cpp ${SOURCE_FILES})
target_link_libraries(main_mem1_low_mut PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(main_mem1_low_mut_MC main_mem1_low_mut_MC.cpp ${SOURCE_FILES})
target_link_libraries(main_mem1_low_mut_MC PRIVATE Eigen3::Eigen lapacke OpenMP::OpenMP_CXX)

add_executable(find_shorter_memory_FR find_shorter_memory_FR.cpp)
