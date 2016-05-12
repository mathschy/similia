project(servers)

add_library(similia_servers_inverted_multi_index_service_lib
  ${PROJECT_SOURCE_DIR}/inverted_multi_index_service.cpp
)

target_link_libraries(similia_servers_inverted_multi_index_service_lib
  similia_common_utils_lib
  similia_proto_lib
  gflags
  glog
  grpc++_unsecure
  grpc
  gpr
  protobuf
)

add_library(similia_servers_similia_service_lib
  ${PROJECT_SOURCE_DIR}/similia_service.cpp
)

target_link_libraries(similia_servers_similia_service_lib
  similia_common_utils_lib
  similia_proto_lib
  similia_utils_features_library_lib
  similia_utils_candidates_finder_lib
  similia_utils_candidates_reranker_lib
  gflags
  glog
  grpc++_unsecure
  grpc
  gpr
  protobuf
)

add_executable(similia_servers_inverted_multi_index_server
  ${PROJECT_SOURCE_DIR}/inverted_multi_index_server.cpp
)

set_target_properties(similia_servers_inverted_multi_index_server
  PROPERTIES OUTPUT_NAME inverted_multi_index_server
)

target_link_libraries(similia_servers_inverted_multi_index_server
  similia_servers_inverted_multi_index_service_lib
  similia_utils_inverted_multi_index_lib
  ${Boost_LIBRARIES}
  gflags
  glog
  grpc++_unsecure
  grpc
  gpr
  protobuf
  pthread
)

add_executable(similia_servers_similia_server
  ${PROJECT_SOURCE_DIR}/similia_server.cpp
)

set_target_properties(similia_servers_similia_server
  PROPERTIES OUTPUT_NAME similia_server
)

target_link_libraries(similia_servers_similia_server
  similia_servers_similia_service_lib
  ${Boost_LIBRARIES}
  gflags
  glog
  grpc++_unsecure
  grpc
  gpr
  protobuf
  pthread
)