# dmrosen 11-Jun-2020

cmake_minimum_required(VERSION 3.1)

# PROJECT CONFIGURATION
project(Preconditioners LANGUAGES C CXX VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # We require C++ 14


# Set build type to 'RelWithDebInfo' if one was not specified by the user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS RelWithDebInfo Release Debug MinSizeRel)
  message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}, as none was specified\n")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode\n")
endif()


# Build the unit tests?
set(BUILD_PRECONDITIONERS_TESTS OFF CACHE BOOL "Build unit tests? [disabled by default]")

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place the libraries built by this project")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

# BUILD CONFIGURATIONS
# Enable faster instruction sets (SIMD/AVX)
set(ENABLE_VECTORIZATION OFF CACHE BOOL "Enable vectorized instruction sets (SIMD/AVX)? [disabled by default]")
# Enable code profiling using gperftools
set(ENABLE_PROFILING OFF CACHE BOOL "Enable code profiling using gperftools")

# Add the .cmake files that ship with Eigen3 to the CMake module path (useful for finding other stuff)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE STRING "The CMake module path used for this project")

if(${ENABLE_VECTORIZATION})
message(STATUS "Enabling SIMD/AVX instruction sets")
add_definitions(-march=native)
endif()

if(${ENABLE_PROFILING})
message(STATUS "Enabling code profiling using Google Performance Tools")
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -lprofiler)
endif()

# Find Eigen library
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library: ${EIGEN3_INCLUDE_DIR}")
endif()

# FIND ADDITIONAL LIBRARIES
# These next operations make use of the .cmake files shipped with Eigen3
find_package(OpenMP REQUIRED)

#Find OPENMP
find_package(OpenMP REQUIRED)

# Find BLAS libraries
find_package(BLAS REQUIRED)

# Find LAPACK libraries
find_package(LAPACK REQUIRED)


# Set location of the include directory for the SYM-ILDL library
set(SymILDL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SymILDL/SymILDL)

# Set the various directories for this project
set(ILDL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Preconditioners/ILDL/include)
set(ILDL_HDR_DIR ${ILDL_INCLUDE_DIR}/ILDL)
set(ILDL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Preconditioners/ILDL/src)

# Get the set of ILDL header and source files
set(ILDL_HDRS
${CMAKE_CURRENT_SOURCE_DIR}/Preconditioners/Types.h
${ILDL_HDR_DIR}/ILDL_utils.h
${ILDL_HDR_DIR}/ILDL.h
)

set(ILDL_SRCS
${ILDL_SOURCE_DIR}/ILDL_utils.cpp
${ILDL_SOURCE_DIR}/ILDL.cpp
)

# The Sym-ILDL headers and Eigen 3 are both referenced by the header files of the ILDL library, hence must be PUBLICLY included (i.e. clients using the SYMILDLSupport headers must also include these headers)
set(ILDL_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${ILDL_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} CACHE INTERNAL "")

add_library(ILDL SHARED ${ILDL_HDRS} ${ILDL_SRCS})
target_include_directories(ILDL PUBLIC ${SymILDL_INCLUDE_DIR} ${ILDL_INCLUDES})
target_link_libraries(ILDL ${BLAS_LIBRARIES})


# Build unit tests
if(${BUILD_PRECONDITIONERS_TESTS})
message(STATUS "Adding unit tests to build")
add_subdirectory(test)
endif()

