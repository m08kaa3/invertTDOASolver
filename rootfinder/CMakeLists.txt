cmake_minimum_required(VERSION 3.20)

# Main library
add_library(RootFinder INTERFACE)
target_link_libraries(RootFinder INTERFACE 
    Eigen3::Eigen)
target_include_directories(RootFinder INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR})