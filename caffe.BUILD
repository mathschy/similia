# This doesn't actually build CAFFE using bazel but it implements the
# includes and linkage that allows to build against caffe, assuming it's been
# built in /opt/caffe

cc_library(
  name = "caffe_includes",
  hdrs = glob(
    [
      "include/**/*.hpp",
      "build/src/**/*.hpp",
      "build/src/**/*.h",
    ]
  ),
  includes = [ "include", "build/src", ],
  visibility = ["//visibility:public"],
  linkopts = [
    "-l/opt/caffe/build/lib/libcaffe.a",
    "-lcblas",
    "-lgflags",
    "-lhdf5_cpp",
    "-lhdf5_hl_cpp",
    "-lhdf5_serial",
    "-lhdf5_serial_hl",
    "-lboost_thread",
  ],
)
