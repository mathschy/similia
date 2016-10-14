project(utils)


add_library(similia_utils_features_utils_lib
  ${PROJECT_SOURCE_DIR}/features_utils.cpp
)

target_link_libraries(similia_utils_features_utils_lib
  ${Boost_LIBRARIES}
)

add_library(similia_utils_features_library_lib
  ${PROJECT_SOURCE_DIR}/features_library.cpp
)

target_link_libraries(similia_utils_features_library_lib
  similia_common_utils_lib
  similia_utils_features_utils_lib
  ${Boost_LIBRARIES}
  protobuf
  pthread
)

add_library(similia_utils_candidates_finder_lib
  ${PROJECT_SOURCE_DIR}/candidates_finder.cpp
)

target_link_libraries(similia_utils_candidates_finder_lib
  similia_utils_features_library_lib
)

add_library(similia_utils_candidates_reranker_lib
  ${PROJECT_SOURCE_DIR}/candidates_reranker.cpp
)

target_link_libraries(similia_utils_candidates_reranker_lib
  similia_utils_candidates_finder_lib
)

add_library(similia_utils_multi_product_quantizer_service_lib
  ${PROJECT_SOURCE_DIR}/multi_product_quantizer_service.cpp
)

target_link_libraries(similia_utils_multi_product_quantizer_service_lib
  similia_common_utils_lib
  similia_proto_lib
  similia_utils_features_utils_lib
  glog
  grpc++_unsecure
  grpc
  protobuf
)

add_library(similia_utils_inverted_multi_index_lib
  ${PROJECT_SOURCE_DIR}/inverted_multi_index.cpp
)

target_link_libraries(similia_utils_inverted_multi_index_lib
  similia_proto_lib
  rocksdb
  jemalloc
  z
  bz2
  snappy
  pthread
)

add_library(similia_utils_cropping_utils_lib
  ${PROJECT_SOURCE_DIR}/cropping_utils.cpp
)

target_link_libraries(similia_utils_cropping_utils_lib
  glog
  opencv_core
)


add_library(similia_utils_features_extractor_lib
  ${PROJECT_SOURCE_DIR}/features_extractor.cpp
)

target_link_libraries(similia_utils_features_extractor_lib
  similia_utils_cropping_utils_lib
  /opt/caffe/build/lib/libcaffe.so
  opencv_core
  opencv_highgui
  opencv_imgproc
  protobuf
)

IF (SIMILIA_CPU_ONLY)
  # activate flag to define CPU_ONLY so that caffe doesn't look for cublas
  target_compile_definitions(similia_utils_features_extractor_lib PUBLIC CPU_ONLY=1)
ENDIF()
# activate flag to have caffe's AddMatVector definition
target_compile_definitions(similia_utils_features_extractor_lib PRIVATE USE_OPENCV=1)
