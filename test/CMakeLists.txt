project(PardisoUtils-tests CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
fetch_googletest(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/googletest
    )


# Enable GTest unit testing
enable_testing()

add_executable(SymILDL_test SymILDL_test.cpp)
target_link_libraries(SymILDL_test SymILDLSupport gtest_main)
