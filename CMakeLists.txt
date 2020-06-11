# dmrosen 11-Jun-2020

cmake_minimum_required(VERSION 3.1)

# PROJECT CONFIGURATION
project(SymILDLSupport LANGUAGES C CXX VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # We require C++ 14

# BUILD CONFIGURATIONS

# Set build type to 'RelWithDebInfo' if one was not specified by the user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS RelWithDebInfo Release Debug MinSizeRel)
  message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}, as none was specified\n")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode\n")
endif()

# Build the unit tests?
set(BUILD_SymILDLSupport_TESTS OFF CACHE BOOL "Build unit tests? [disabled by default]")
# Enable faster instruction sets (SIMD/AVX)?
set(ENABLE_VECTORIZATION OFF CACHE BOOL "Enable vectorized instruction sets (SIMD/AVX)? [disabled by default]")
# Enable code profiling using gperftools
set(ENABLE_PROFILING OFF CACHE BOOL "Enable code profiling using gperftools [disabled by default]")

# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

# Add the .cmake files that ship with Eigen3 to the CMake module path (useful for finding other stuff)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE STRING "The CMake module path used for this project")


# PardisoSupport DEPENDENCIES

# Find Eigen library
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library: ${EIGEN3_INCLUDE_DIR}")
endif()

# Find BLAS libraries
find_package(BLAS REQUIRED)

# Find LAPACK libraries
find_package(LAPACK REQUIRED)


if(${ENABLE_VECTORIZATION})
message(STATUS "Enabling SIMD/AVX instruction sets")
add_definitions(-march=native)
endif()

if(${ENABLE_PROFILING})
message(STATUS "Enabling code profiling using Google Performance Tools")
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -lprofiler)
endif()


# SET UP SymILDLSupport LIBRARY

# Set the various directories for this project
set(SymILDLSupport_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SymILDLSupport/include)
set(SymILDLSupport_HDR_DIR ${SymILDLSupport_INCLUDE_DIR}/SymILDLSupport)
set(SymILDLSupport_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SymILDLSupport/src)

# Get the set of SymILDLSupport header and source files
set(SymILDLSupport_HDRS
${SymILDLSupport_HDR_DIR}/SymILDLSupport_types.h
${SymILDLSupport_HDR_DIR}/SymILDLUtils.h
#${SymILDLSupport_HDR_DIR}/ILDLFactorization.h
)

set(SymILDLSupport_SRCS
${SymILDLSupport_SOURCE_DIR}/SymILDLUtils.cpp
#${SymILDLSupport_SOURCE_DIR}/ILDLFactorization.cpp
)


# The SYMILDLSupport headers and Eigen 3 are all referenced by the header files of the SYMILDLSupport library, hence must be PUBLICLY included (i.e. clients using the SYMILDLSupport headers must also include these headers)
set(SymILDLSupport_INCLUDES ${SymILDLSupport_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} CACHE INTERNAL "")


# Build the SYMILDLSupport library
add_library(${PROJECT_NAME} SHARED ${SymILDLSupport_HDRS} ${SymILDLSupport_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC ${SymILDLSupport_INCLUDES})
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Build unit tests
if(${BUILD_SymILDLSupport_TESTS})
message(STATUS "Adding unit tests to build")
add_subdirectory(test)
endif()

# EXPORT SymILDLSupport LIBRARY

# Add add entry for this project into CMake's package registry, so that this project can be found by other CMake projects
export(PACKAGE ${PROJECT_NAME})
# Create a configuration file for this project, so that it can be imported by other CMake projects
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

