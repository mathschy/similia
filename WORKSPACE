new_local_repository(
  name = "caffe",
  path = "/opt/caffe",
  build_file = "caffe.BUILD",
)

new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)

git_repository(
    name   = "com_github_gflags_gflags",
    commit = "9314597d4b742ed6f95665241345e590a0f5759b",
    remote = "https://github.com/gflags/gflags.git",
)

bind(
    name = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)
