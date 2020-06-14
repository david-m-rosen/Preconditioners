project(PardisoUtils-tests CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
fetch_googletest(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/googletest
    )


# Enable GTest unit testing
enable_testing()

add_executable(ILDLFactorizationTest ILDLFactorizationTest.cpp)
target_link_libraries(ILDLFactorizationTest SymILDLSupport gtest_main)

add_executable(example example.cpp)
target_link_libraries(example SymILDLSupport)

add_executable(minimal_working_example minimal_working_example.cpp)
target_link_libraries(minimal_working_example SymILDLSupport)
