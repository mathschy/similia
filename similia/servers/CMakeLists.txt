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