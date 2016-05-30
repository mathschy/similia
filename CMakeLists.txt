project(similia)
cmake_minimum_required(VERSION 2.8)

set(PROJECT_INCLUDE_DIR .)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

include_directories("${PROJECT_INCLUDE_DIR}")

# C++ 11
SET(CMAKE_CXX_FLAGS "-O3")
add_definitions(
  -std=c++11
)

# Boost
find_package(Boost 1.55 COMPONENTS system filesystem REQUIRED)

# GTest
add_subdirectory(third_party/googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


add_subdirectory(similia)
