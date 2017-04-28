cc_binary(
  name = "inverted_multi_index_server",
  srcs = [
    "inverted_multi_index_server.cpp",
  ],
  deps = [
    ":inverted_multi_index_service",
    "//external:gflags",
    "//similia/utils:inverted_multi_index",
  ],
  linkopts = [
    "-lglog",
  ],
)

cc_library(
  name = "inverted_multi_index_service",
  srcs = [
    "inverted_multi_index_service.cpp",
  ],
  hdrs = [
    "inverted_multi_index_service.h",
  ],
  deps = [
    "//similia/common/utils",
    "//similia/proto",
    "//similia/utils:inverted_multi_index",
  ],
  visibility = ["//visibility:public"],
)

cc_binary(
  name = "similia_server",
  srcs = [
    "similia_server.cpp",
  ],
  deps = [
    ":similia_service",
    "//external:gflags",
  ],
  linkopts = [
    "-lglog",
  ],
)

cc_library(
  name = "similia_service",
  srcs = [
    "similia_service.cpp",
  ],
  hdrs = [
    "similia_service.h",
  ],
  deps = [
    "//similia/common/utils",
    "//similia/proto",
    "//similia/utils:candidates_finder",
    "//similia/utils:candidates_reranker",
    "//similia/utils:features_library",
    "//similia/utils:matrix_utils",
  ],
  visibility = ["//visibility:public"],
)
