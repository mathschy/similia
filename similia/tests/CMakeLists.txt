project(tests)


add_executable(similia_tests_inverted_multi_index_test
  ${PROJECT_SOURCE_DIR}/inverted_multi_index_test.cpp
)

target_link_libraries(similia_tests_inverted_multi_index_test
  similia_utils_inverted_multi_index_lib
  similia_proto_lib
  ${Boost_LIBRARIES}
  gflags
  glog
  protobuf
  gtest
  gtest_main
  pthread
)

set_target_properties(similia_tests_inverted_multi_index_test
  PROPERTIES OUTPUT_NAME inverted_multi_index_test
)

add_test(NAME inverted_multi_index_test
  COMMAND $<TARGET_FILE:similia_tests_inverted_multi_index_test>
)