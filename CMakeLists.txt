cmake_minimum_required(VERSION 3.20)
project(SKPD C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DN=130 -DREPETITIONS=3 -DDEBUG")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
set(OMP_EXE_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_executable(source source.c)
target_link_libraries(source PRIVATE OpenMP::OpenMP_C)
set_target_properties(source PROPERTIES COMPILE_FLAGS ${OMP_EXE_FLAGS})

add_executable(jac_3d jac_3d.c)
target_link_libraries(jac_3d PRIVATE OpenMP::OpenMP_C)
set_target_properties(jac_3d PROPERTIES COMPILE_FLAGS ${OMP_EXE_FLAGS})

add_executable(jac_3d_omp jac_3d_omp.c)
target_link_libraries(jac_3d_omp PRIVATE OpenMP::OpenMP_C)
set_target_properties(jac_3d_omp PROPERTIES COMPILE_FLAGS ${OMP_EXE_FLAGS})

add_executable(jac_3d_mpi jac_3d_mpi.c)
target_link_libraries(jac_3d_mpi ${MPI_LIBRARIES})

include_directories(${MPI_INCLUDE_PATH})
