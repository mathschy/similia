cc_library(
  name = "proto",
  srcs = [
    "similia.grpc.pb.cc",
    "similia.pb.cc",
  ],
  hdrs = [
    "compressed_elements_generated.h",
    "compressing_ids_generated.h",
    "similia.grpc.pb.h",
    "similia.pb.h",
  ],
  deps = [
  ],
  linkopts = [ "-lgrpc++_unsecure -lgrpc -lgpr -lprotobuf" ],
  visibility = ["//visibility:public"],
)
