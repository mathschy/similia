project(proto)

add_library(similia_proto_lib
  ${PROJECT_SOURCE_DIR}/similia.grpc.pb.cc
  ${PROJECT_SOURCE_DIR}/similia.pb.cc
)

target_link_libraries(similia_proto_lib
  grpc++_unsecure
  grpc
  gpr
  protobuf
)