cc_library(
  name = "utils",
  srcs = [
    "file_utils.cpp",
    "metrics.cpp",
  ],
  hdrs = [
    "base.h",
    "file_utils.h",
    "metrics.h",
  ],
  deps = [
  ],
  linkopts = [
    "-lpthread",
    "-lboost_context",
    "-lboost_system",
    "-lprotobuf",
  ],
  visibility = ["//visibility:public"],
)
