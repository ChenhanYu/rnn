# Check cmake version.
cmake_minimum_required (VERSION 2.8)


# Project specification.
project (GSKNN)
set (GSKNN_VERSION_MAJOR 1)
set (GSKNN_VERSION_MINOR 1)
set (GSKNN_VERSION_PATCH 0)
set (GSKNN_VERSION ${GSKNN_VERSION_MAJOR}.${GSKNN_VERSION_MINOR}.${GSKNN_VERSION_PATCH})


# Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")


# Turn on testing.
ENABLE_TESTING()


# Decide architecture
set (GSKNN_ARCH_MAJOR $ENV{GSKNN_ARCH_MAJOR})
set (GSKNN_ARCH_MINOR $ENV{GSKNN_ARCH_MINOR})
set (GSKNN_ARCH ${GSKNN_ARCH_MAJOR}/${GSKNN_ARCH_MINOR})


# Compiler Options (GSKNN must use Intel compiler.)
set (CMAKE_C_COMPILER   icc )
set (CMAKE_CXX_COMPILER icpc)
set (GSKNN_CFLAGS          "-O3 -openmp -mavx -Werror -Wall -pedantic")
set (CMAKE_C_FLAGS      "${CMAKE_C_FLAGS} ${GSKNN_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${GSKNN_CFLAGS}")


# Software dependencies.
set (MKL_DIR $ENV{GSKNN_MKL_DIR})


# Linking infos.
set (CMAKE_EXE_LINKER_FLAGS "-lpthread -lm -mkl=parallel")


# Headers.
include_directories ("include" "${CMAKE_SOURCE_DIR}/kernels/${GSKNN_ARCH}" "${MKL_DIR}/include")
file (GLOB GSKNN_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${GSKNN_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
file (GLOB FRAME_CC_SRC  ${CMAKE_SOURCE_DIR}/frame/*.c)
file (GLOB FRAME_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernels/${GSKNN_ARCH}/*.c)


# Build the static library.
add_library (rnn ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC})


# Build the executable files.
add_executable (test_dgsrnn_stl.x ${CMAKE_SOURCE_DIR}/test/test_dgsrnn_stl.cpp)
target_link_libraries(test_dgsrnn_stl.x rnn)
add_executable (test_dgsrnn.x ${CMAKE_SOURCE_DIR}/test/test_dgsrnn.c)
target_link_libraries(test_dgsrnn.x rnn)


# Install shell script
file (GLOB GSKNN_SCRIPTS ${CMAKE_SOURCE_DIR}/test/*.sh)
install (FILES ${GSKNN_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)


# Report
message ("===================================================")
message ("Version      =${GSKNN_VERSION}")
message ("Source       =${CMAKE_SOURCE_DIR}")
message ("Target       =${CMAKE_BINARY_DIR}")
message ("KS_ARCH      =${GSKNN_ARCH}")
message ("CC           =${CMAKE_C_COMPILER}")
message ("CFLAGS       =${CMAKE_C_FLAGS}")
message ("CXX          =${CMAKE_CXX_COMPILER}")
message ("CXXFLAGS     =${CMAKE_CXX_FLAGS}")
message ("Linker       =${CMAKE_EXE_LINKER_FLAGS}")
message ("MKLDIR       =${MKL_DIR}")
message ("===================================================")
