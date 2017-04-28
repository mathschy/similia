#include "similia/utils/features_extractor.h"

#include <chrono>

#include <boost/algorithm/string.hpp>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <opencv2/opencv.hpp>

namespace similia {

using std::chrono::steady_clock;

using boost::shared_ptr;
using caffe::Blob;
using caffe::Caffe;
using caffe::MemoryDataLayer;
using caffe::Net;

// Helper
cv::Mat DecodeImage(const std::string& image) {
  cv::Mat img;
  try {
    // read an image from memory buffer and force conversion to 3 channels.
    img = cv::imdecode(cv::Mat(std::vector<char>(image.begin(), image.end())), CV_LOAD_IMAGE_COLOR);
    LOG(INFO) << "image decoded, size : " << img.cols << "x" << img.rows;
  } catch (const cv::Exception& ex) {
    LOG(ERROR) << "couldn't imdecode (possibly corrupt data, possibly image not found?). "
        << ex.what();
  }
  return img;
}

// constructor
FeaturesExtractor::FeaturesExtractor(const std::string& path_to_caffe_model_weights,
                                     const std::string& path_to_deploy_prototxt,
                                     const std::string& blob_names,
                                     const int gpu) {
  if (gpu >= 0) {
    LOG(INFO) << "Querying GPU devices...";
    Caffe::DeviceQuery();
    LOG(INFO) << "Device id " << gpu << " status: " << Caffe::CheckDevice(gpu);
    LOG(INFO) << "Activating GPU mode with device ID = " << gpu;
    Caffe::SetDevice(gpu);
    Caffe::set_mode(Caffe::GPU);
    LOG(INFO) << "Caffe is in GPU mode.";
    Caffe::DeviceQuery();
    gpu_ = gpu;
  } else {
    LOG(INFO) << "Caffe is in CPU mode.";
  }
  // set loglevel to 1 to avoid a bunch of INFO messages when initiliazing the net.
  LOG(INFO) << "Don't log INFO messages during net initialiation...";
  int min_log_level = FLAGS_minloglevel;
  FLAGS_minloglevel = 1;
  this->net_ = std::unique_ptr<Net<float>>(new Net<float>(path_to_deploy_prototxt, caffe::TEST));
  FLAGS_minloglevel = min_log_level;  // set minloglevel back to what it was.
  LOG(INFO) << "Net initialized. Back to previous minloglevel.";
  this->net_->CopyTrainedLayersFrom(path_to_caffe_model_weights);
  boost::split(blob_names_, blob_names, boost::is_any_of(","));
}

std::vector<float> FeaturesExtractor::CropAndExtractFeatures(const std::string& image) {
  try {
    return ExtractFeatures(image, ComputeCropBounds(image));
  } catch (const cv::Exception& ex) {
    LOG(ERROR) << "couldn't extract features or crop bounds: " << ex.what();
    return {};
  }
}

std::vector<float> FeaturesExtractor::ExtractFeatures(const std::string& image) {
  return ExtractFeatures(DecodeImage(image));
}


std::vector<float> FeaturesExtractor::ExtractFeatures(const std::string& image, const CropBounds& crop_bounds) {
  cv::Mat img = DecodeImage(image);
  try {
    img = CropImage(img, crop_bounds);
    LOG(INFO) << "image cropped, new size: " << img.cols << "x" << img.rows;
  } catch (const cv::Exception& ex) {
    LOG(ERROR) << "couldn't crop image (possibly corrupt data, possibly image not found?). "
        << ex.what();
    return {};
  }
  return ExtractFeatures(img);
}

std::vector<float> FeaturesExtractor::ExtractFeatures(const cv::Mat& img) {
  if (img.empty()) {
    LOG(ERROR) << "image is empty";
    return {};
  }
  CHECK_EQ(img.channels(), 3) << "this should not happen, we should obtain 3 channels with CV_LOAD_IMAGE_COLOR";

  steady_clock::time_point begin_extract_features = steady_clock::now();

  // For some reason the gpu mode is not persistent when we use this in a gRPC server.
  // We have to reactivate gpu mode.
  if (gpu_ >= 0 && Caffe::mode() != Caffe::GPU) {
    LOG(INFO) << "Reactivating GPU mode with device ID = " << gpu_;
    Caffe::SetDevice(gpu_);
    Caffe::set_mode(Caffe::GPU);
    CHECK(Caffe::mode() == Caffe::GPU);
  }

  cv::Mat img_resized;
  try {
    // We resize to 224 because it is the input size of GoogLeNet.
    // Thus we will take the whole image as input.
    cv::resize(img, img_resized, cv::Size(224,224), 0, 0, cv::INTER_LINEAR);
  } catch (const cv::Exception& ex) {
    // This is probably mostly due to corrupt downloads. We might want to retry
    // it if we start seeing too many of those.
    LOG(ERROR) << "cannot resize image: " << ex.what();
    return {};
  }

  size_t num_extracted_layers = blob_names_.size(); // number of layers to extract

  for (size_t i = 0; i < num_extracted_layers; ++i) {
    CHECK(this->net_->has_blob(blob_names_[i]))
    << "Unknown feature blob name " << blob_names_[i]
        << " in the network " << this->path_to_deploy_prototxt_;
  }

  // this is generic code but we use it only one image at a time for now.
  int num_mini_batches = 1;
  std::vector<float> result;
  std::vector<int> image_indices(num_extracted_layers, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    std::vector<cv::Mat> images(1, img_resized);
    std::vector<int> labels(1, 0);
    shared_ptr<MemoryDataLayer<float>> md_layer =
        boost::dynamic_pointer_cast<MemoryDataLayer<float>>(this->net_->layers()[0]);
    md_layer->AddMatVector(images, labels);

    VLOG(1) << "Forwarding";
    net_->Forward();
    VLOG(1) << "Forwarded";
    LOG(INFO) << "num_extracted_layers = " << num_extracted_layers;
    for (std::size_t i = 0; i < num_extracted_layers; ++i) { // only 1 for us
      const shared_ptr<Blob<float>> feature_blob = net_->blob_by_name(blob_names_[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const float* feature_blob_data;
      LOG(INFO) << "batch_size = " << batch_size;
      for (int n = 0; n < batch_size; ++n) {
        feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
        LOG(INFO) << "dim_features = " << dim_features;
        for (int d = 0; d < dim_features; ++d) {
          result.push_back(feature_blob_data[d]);
        }
      }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int i = 0; i < num_extracted_layers; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  steady_clock::time_point end_extract_features = steady_clock::now();
  LOG(INFO) << "ExtractFeatures took: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end_extract_features-begin_extract_features).count()
      << " ms.";
  return result;
}

}  // namespace similia
