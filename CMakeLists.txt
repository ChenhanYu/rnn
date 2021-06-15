# Check cmake version.
cmake_minimum_required (VERSION 3.5)


# Project specification.
project (GSKNN C CXX)
set (GSKNN_VERSION_MAJOR 1)
set (GSKNN_VERSION_MINOR 1)
set (GSKNN_VERSION_PATCH 0)
set (GSKNN_VERSION ${GSKNN_VERSION_MAJOR}.${GSKNN_VERSION_MINOR}.${GSKNN_VERSION_PATCH})


find_package(BLAS)

# Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")

#set(SUPPORTED_PLATFORM true)
#if(not CMAKE_SYSTEM_PROCESSOR)
#  message(WARNING "Target processor architecture is not specified.")
#  set(SUPPORTED_PLATFORM false)
#elseif(not CMAKE_SYSTEM_PROCESSOR matches "^(AMD64|x86_64|armv7|armv7f|armv7s|armv7k|arm64|aarch64)$")
#  message(WARNING "Target processor architecture ${CMAKE_SYSTEM_PROCESSOR} is not supported in GSKNN.")
#  set(SUPPORTED_PLATFORM false)
#endif()
  

# Turn on testing.
ENABLE_TESTING()


# Decide architecture
set (GSKNN_ARCH_MAJOR $ENV{GSKNN_ARCH_MAJOR})
set (GSKNN_ARCH_MINOR $ENV{GSKNN_ARCH_MINOR})
set (GSKNN_ARCH ${GSKNN_ARCH_MAJOR}/${GSKNN_ARCH_MINOR})

message("Compiler ID: ${CMAKE_C_COMPILER_ID}")
# Compiler Options.
if (${CMAKE_C_COMPILER_ID} MATCHES "Intel")
  #set (CMAKE_C_COMPILER   icc )
  #set (CMAKE_CXX_COMPILER icpc)
  #set (GSKNN_CFLAGS          "-O2 -qopenmp -mfloat-abi=hard -mfpu=neon -march=armv7-a")
  set (GSKNN_CFLAGS          "-O2 -qopenmp -mavx")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -qopenmp")
else (${CMAKE_C_COMPILER_ID} MATCHES "GNU")
  #set (CMAKE_C_COMPILER   gcc)
  #set (CMAKE_CXX_COMPILER g++)
  #set (GSKNN_CFLAGS          "-O2 -fopenmp -mfloat-abi=hard -mfpu=neon -march=armv7-a")
  set (GSKNN_CFLAGS          "-O2 -fopenmp -mavx")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp")
endif ()

find_package(BLAS)

if (BLAS_FOUND)
  link_libraries(${BLAS_LIBRARIES})
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  set (GSKNN_CFLAGS          "${GSKNN_CFLAGS} -DUSE_BLAS")
endif()

set (CMAKE_C_FLAGS      "${CMAKE_C_FLAGS} ${GSKNN_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${GSKNN_CFLAGS}")


# Software dependencies.
#set (MKL_DIR $ENV{GSKNN_MKL_DIR})

# Headers.
include_directories ("include" "${CMAKE_SOURCE_DIR}/kernels/${GSKNN_ARCH}" "${MKL_DIR}/include")
file (GLOB GSKNN_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${GSKNN_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
file (GLOB FRAME_CC_SRC  ${CMAKE_SOURCE_DIR}/frame/*.c)
file (GLOB FRAME_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernels/${GSKNN_ARCH}/*.c)


# Build the static library.
add_library (gsknn ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC})

# Build the executable files.
add_executable (test_sgsknn.x ${CMAKE_SOURCE_DIR}/test/test_sgsknn.c)
target_link_libraries(test_sgsknn.x gsknn m)
#target_link_libraries(test_sgsknn.x openblas)
add_executable (test_sgsknn_stl.x ${CMAKE_SOURCE_DIR}/test/test_sgsknn_stl.cpp)
target_link_libraries(test_sgsknn_stl.x gsknn m)
#target_link_libraries(test_sgsknn_stl.x openblas)
add_executable (test_dgsknn.x ${CMAKE_SOURCE_DIR}/test/test_dgsknn.c)
target_link_libraries(test_dgsknn.x gsknn m)
#target_link_libraries(test_dgsknn.x openblas)
add_executable (test_dgsknn_stl.x ${CMAKE_SOURCE_DIR}/test/test_dgsknn_stl.cpp)
target_link_libraries(test_dgsknn_stl.x gsknn m)
#target_link_libraries(test_dgsknn_stl.x openblas)


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
message ("MKLDIR       =${BLAS_LIBRARIES}")
message ("===================================================")
