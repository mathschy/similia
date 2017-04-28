cc_binary(
  name = "extract_features",
  srcs = [
    "extract_features.cpp",
  ],
  deps = [
    "//similia/common/utils",
    "//similia/utils:features_extractor",
    "//similia/utils:features_utils",
  ],
  linkopts = [
    "-lglog",
    "-lgflags",
  ],
)

cc_binary(
  name = "inverted_multi_index_test_client",
  srcs = [
    "inverted_multi_index_test_client.cpp",
  ],
  deps = [
    "//external:gflags",
    "//similia/proto",
  ],
  linkopts = [
    "-lglog",
  ],
)

cc_binary(
  name = "similia_test_client",
  srcs = [
    "similia_test_client.cpp",
  ],
  deps = [
    "//external:gflags",
    "//similia/common/utils",
    "//similia/proto",
    "//similia/utils:cropping_utils",
    "//similia/utils:features_extractor",
  ],
  linkopts = [
    "-lglog",
  ],
)
