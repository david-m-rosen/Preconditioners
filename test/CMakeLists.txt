#project(PardisoUtils-tests CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
fetch_googletest(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/googletest
    )


# Enable GTest unit testing
enable_testing()

add_executable(ILDLTest ILDLTest.cpp)
target_link_libraries(ILDLTest ILDL gtest_main)
