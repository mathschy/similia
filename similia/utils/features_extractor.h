#pragma once

#include <memory>
#include <string>
#include <vector>

#include "caffe/net.hpp"

#include "similia/utils/cropping_utils.h"


namespace similia {

class FeaturesExtractor {
 public:
  FeaturesExtractor(const std::string& path_to_caffe_model_weights,
                    const std::string& path_to_deploy_prototxt,
                    const std::string& blob_names,
                    const int gpu);

  // Extracts and returns the features for a given image. 'image' must be the raw image data,
  std::vector<float> ExtractFeatures(const std::string& image);

  // Same as above but after cropping it with the given crop_bounds.
  std::vector<float> ExtractFeatures(const std::string& image, const CropBounds& crop_bounds);

  // Same as above but compute the crop_bounds.
  std::vector<float> CropAndExtractFeatures(const std::string& image);


 private:
  std::vector<float> ExtractFeatures(const cv::Mat& image);

  std::unique_ptr<caffe::Net<float>> net_;
  std::vector<std::string> blob_names_;
  std::string path_to_caffe_model_weights_;
  std::string path_to_deploy_prototxt_;
  int gpu_{-1};

};
} // namespace similia
