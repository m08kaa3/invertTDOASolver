cmake_minimum_required(VERSION 3.10)
project(TestTask)
set (CMAKE_CXX_STANDARD 20)

add_subdirectory(eigen)
add_subdirectory(rootfinder)

add_executable(main InvertTDOAsolver.cpp main.cpp)
target_link_libraries(main PRIVATE Eigen3::Eigen RootFinder)