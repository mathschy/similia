// This file defines a tool to extract features for all images in a given directory. It stores them in a csv file.
//
// Sample usage:
// ./build/extract_features --logtostderr --images_dir images_directory --output_file features_file.csv
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <similia/common/utils/file_utils.h>
#include <similia/utils/features_extractor.h>
#include <similia/utils/features_utils.h>

using boost::filesystem::directory_iterator;
using boost::filesystem::path;

DEFINE_string(images_dir,
              "",
              "Path to the directory that contains images from which to extract features.");
DEFINE_string(output_file,
              "",
              "Path to which output the results.");
DEFINE_string(weights,
              "similia/data/bvlc_googlenet.caffemodel",
              "Path to the .caffemodel file.");
DEFINE_string(deploy_prototxt,
              "similia/data/deploy_mc.prototxt",
              "Path to the deploy .prototxt file.");
DEFINE_string(blob_names, "pool5/7x7_s1", "Name of the blob to extract features from.");
DEFINE_int32(gpu, -1, "Run in GPU mode on given device ID. -1 means run on CPU.");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  // trick to check that we can write to output_file before doing all the processing.
  common_utils::WriteToFileOrDie(FLAGS_output_file, "");

  // list all files in the images dir
  path images_path(FLAGS_images_dir);
  CHECK(boost::filesystem::is_directory(images_path)) << ": " << FLAGS_images_dir << " is not a directory.";
  std::vector<path> image_paths;
  std::copy(directory_iterator(images_path), directory_iterator(), std::back_inserter(image_paths));
  LOG(INFO) << "There are " << image_paths.size() << " files in folder " << images_path;
  // we sort the paths to guarantee we have the same order.
  std::sort(image_paths.begin(), image_paths.end());

  similia::FeaturesExtractor fe(FLAGS_weights, FLAGS_deploy_prototxt, FLAGS_blob_names, FLAGS_gpu);

  std::ofstream output_stream(FLAGS_output_file);
  int num_extracted = 0;
  for (const path& image_path : image_paths) {
    std::vector<float> features = fe.CropAndExtractFeatures(common_utils::ReadFromFileOrDie(image_path));
    CHECK_EQ(similia::kFeatureDimensions, features.size());
    output_stream << features[0];
    for (int i = 1; i < similia::kFeatureDimensions; ++i) {
      output_stream << ", " << features[i];
    }
    output_stream << std::endl;
    ++num_extracted;
    LOG(INFO) << "extracted features for " << num_extracted << " images.";
  }
  output_stream.close();
  return 0;
}