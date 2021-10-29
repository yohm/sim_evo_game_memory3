cmake_minimum_required(VERSION 3.7)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_OSX_DEPLOYMENT_TARGET 11.6)

#set(CMAKE_VERBOSE_MAKEFILE 1)

set(SOURCE_FILES Action.cpp Action.hpp DirectedGraph.hpp DirectedGraph.cpp
        UnionFind.hpp Partition.hpp StrategyM3.hpp StrategyM3.cpp)
find_package(MPI)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
# See https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html

include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/json/include ${CMAKE_SOURCE_DIR}/eigen)
link_directories(/usr/local/lib)

add_executable(test_DirectedGraph test_DirectedGraph.cpp DirectedGraph.hpp DirectedGraph.cpp)

add_executable(test_StrategyM3 test_StrategyM3.cpp ${SOURCE_FILES})
target_link_libraries(test_StrategyM3 PRIVATE Eigen3::Eigen)

add_executable(test_StrategySpace test_StrategySpace.cpp StrategySpace.hpp)
target_link_libraries(test_StrategySpace PRIVATE Eigen3::Eigen)

add_executable(test_Scalapack test_Scalapack.cpp Scalapack.cpp)
target_link_libraries(test_Scalapack PRIVATE MPI::MPI_CXX)
target_link_libraries(test_Scalapack PRIVATE scalapack)

add_executable(main_evo main_evo.cpp ${SOURCE_FILES})
target_link_libraries(main_evo PRIVATE Eigen3::Eigen)
target_link_libraries(main_evo PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(main_evo PRIVATE EIGEN_DONT_PARALLELIZE)

add_executable(main_evo_MPI main_evo_MPI.cpp ${SOURCE_FILES} Scalapack.hpp Scalapack.cpp)
target_link_libraries(main_evo_MPI PRIVATE Eigen3::Eigen)
target_link_libraries(main_evo_MPI PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(main_evo_MPI PRIVATE EIGEN_DONT_PARALLELIZE)
target_link_libraries(main_evo_MPI PRIVATE MPI::MPI_CXX)
target_link_libraries(main_evo_MPI PRIVATE scalapack)

add_executable(test_MultiLevelEvoGame test_MultiLevelEvoGame.cpp MultiLevelEvoGame.hpp ${SOURCE_FILES})
target_link_libraries(test_MultiLevelEvoGame PRIVATE Eigen3::Eigen)
target_compile_definitions(test_MultiLevelEvoGame PRIVATE EIGEN_DONT_PARALLELIZE)
target_link_libraries(test_MultiLevelEvoGame PRIVATE OpenMP::OpenMP_CXX)

add_executable(main_multi_evo_parallel main_multi_evo_parallel.cpp ${SOURCE_FILES} MultiLevelEvoGame.hpp)
target_link_libraries(main_multi_evo_parallel PRIVATE Eigen3::Eigen)
target_link_libraries(main_multi_evo_parallel PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(main_multi_evo_parallel PRIVATE EIGEN_DONT_PARALLELIZE)
