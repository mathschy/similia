#ifndef SIMILIA_UTILS_CROPPING_UTILS_H
#define SIMILIA_UTILS_CROPPING_UTILS_H

#include <opencv2/opencv.hpp>


namespace similia {
struct CropBounds {
  int up;
  int bottom;
  int left;
  int right;
};

// Functions for detecting the external borders in the image and crop them
CropBounds CannyGetCropBounds(const cv::Mat& src, int lower_thresh, int upper_thresh, int kernel_size);
cv::Mat CropImage(const cv::Mat& src, const CropBounds& crop_bounds);
CropBounds ComputeCropBounds(const std::string& image);
}  // namespace similia

#endif // SIMILIA_UTILS_CROPPING_UTILS_H
