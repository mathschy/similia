cc_library(
  name = "candidates_finder",
  srcs = [
    "candidates_finder.cpp",
  ],
  hdrs = [
    "candidates_finder.h",
  ],
  deps = [
    ":features_library",
    ":features_utils",
    "//similia/common/utils",
    "//similia/proto",
  ],
  linkopts = [ "-lboost_filesystem -lglog" ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "candidates_reranker",
  srcs = [
    "candidates_reranker.cpp",
  ],
  hdrs = [
    "candidates_reranker.h",
  ],
  deps = [
    ":candidates_finder",
    ":matrix_utils",
    "//similia/common/utils",
    "//similia/proto",
  ],
  linkopts = [ "-lboost_filesystem -lglog" ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "cropping_utils",
  srcs = [
    "cropping_utils.cpp",
  ],
  hdrs = [
    "cropping_utils.h",
  ],
  deps = [
  ],
  linkopts = [ "-lopencv_core" ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "features_extractor",
  srcs = [
    "features_extractor.cpp",
  ],
  hdrs = [
    "features_extractor.h",
  ],
  deps = [
    ":cropping_utils",
    "@caffe//:caffe_includes",
  ],
  copts= [
    "-DCPU_ONLY=1",
    "-DUSE_OPENCV=1",
  ],
  linkopts = [
    "-lboost_filesystem",
    "-lglog",
    "-lopencv_core",
    "-lopencv_highgui",
    "-lopencv_imgproc"
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "features_library",
  srcs = [
    "features_library.cpp",
  ],
  hdrs = [
    "features_library.h",
  ],
  deps = [
    ":features_utils",
    "//similia/common/utils",
    "//similia/proto",
  ],
  linkopts = [
    "-lboost_filesystem",
    "-lglog",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "features_utils",
  srcs = [
    "features_utils.cpp",
  ],
  hdrs = [
    "features_utils.h",
  ],
  deps = [
  ],
  linkopts = [
    "-lboost_filesystem",
    "-lboost_system",
    "-lglog",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "inverted_multi_index",
  srcs = [
    "inverted_multi_index.cpp",
  ],
  hdrs = [
    "inverted_multi_index.h",
  ],
  deps = [
    ":features_utils",
    "//similia/common/utils",
    "//similia/proto",
  ],
  linkopts = [
    "-lboost_filesystem",
    "-lglog",
    "-lrocksdb",
    "-ljemalloc",
    "-llz4",
    "-lz",
    "-lbz2",
    "-lsnappy",
    "-lpthread",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "matrix_utils",
  hdrs = [
    "matrix_utils.h",
  ],
  deps = [
    "//similia/common/utils",
    "//similia/proto",
  ],
  linkopts = [ "-lboost_filesystem" ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "multi_product_quantizer_service",
  srcs = [
    "multi_product_quantizer_service.cpp",
  ],
  hdrs = [
    "multi_product_quantizer_service.h",
  ],
  deps = [
    ":features_utils",
    ":matrix_utils",
    "//similia/common/utils",
    "//similia/proto",
  ],
  linkopts = [ "-lglog" ],
  visibility = ["//visibility:public"],
)
