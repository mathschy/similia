project(similia)
cmake_minimum_required(VERSION 2.8)

set(PROJECT_INCLUDE_DIR .)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

include_directories("${PROJECT_INCLUDE_DIR}")

SET(CMAKE_CXX_FLAGS "-O3")
add_definitions(
  -std=c++11
)

IF (DEFINED SIMILIA_CPU_ONLY)
  add_definitions(-DCPU_ONLY)
ENDIF()

add_subdirectory(similia)
