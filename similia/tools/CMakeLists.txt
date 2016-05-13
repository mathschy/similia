project(tools)


add_executable(similia_tools_similia_test_client
  ${PROJECT_SOURCE_DIR}/similia_test_client.cpp
)

set_target_properties(similia_tools_similia_test_client
  PROPERTIES OUTPUT_NAME similia_test_client
)

target_link_libraries(similia_tools_similia_test_client
  similia_common_utils_lib
  similia_proto_lib
  similia_utils_features_extractor_lib
  ${Boost_LIBRARIES}
  gflags
  glog
  grpc++_unsecure
  grpc
  gpr
  protobuf
  pthread
)