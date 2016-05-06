project(common)
cmake_minimum_required(VERSION 2.8)

find_package(Boost 1.55 COMPONENTS thread program_options system filesystem REQUIRED)

add_library(similia_common_utils_lib
  ${PROJECT_SOURCE_DIR}/utils/file_utils.cpp
  ${PROJECT_SOURCE_DIR}/utils/metrics.cpp
)

target_link_libraries(similia_common_utils_lib
  ${Boost_LIBRARIES}
)