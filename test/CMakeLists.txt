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

add_executable(LSCholTest LSCholTest.cpp)
target_link_libraries(LSCholTest /usr/lib/x86_64-linux-gnu/libcholmod.so LSChol gtest_main)
