#include <similia/utils/cropping_utils.h>

namespace similia {

CropBounds CannyGetCropBounds(const cv::Mat& src, int lower_thresh, int upper_thresh, int kernel_size) {
  cv::Mat src_gray;
  cv::Mat edges;

  /// Convert the image to grayscale
  cv::cvtColor(src, src_gray, CV_RGB2GRAY);
  /// Smooth the image
  cv::GaussianBlur(src_gray, src_gray, cv::Size(kernel_size, kernel_size), 0, 0);
  /// Canny detector
  cv::Canny(src_gray, edges, lower_thresh, upper_thresh, kernel_size);

  int l, r, u, b;
  int w = src_gray.cols, h = src_gray.rows;

  /// Find borders
  // left limit
  for (l = 0; l < w; l++) {
    int sum = 0;
    for (int j = 0; j < h; j++) {
      sum += edges.at<uchar>(j, l);
    }
    if (sum > 0) {
      break;
    }
  }

  // right limit
  for (r = w - 1; r >= 0; r--) {
    int sum = 0;
    for (int j = 0; j < h; j++) {
      sum += edges.at<uchar>(j, r);
    }
    if (sum > 0) {
      break;
    }
  }

  // upper limit
  for (u = 0; u < h; u++) {
    int sum = 0;
    for (int j = 0; j < w; j++) {
      sum += edges.at<uchar>(u, j);
    }
    if (sum > 0) {
      break;
    }
  }

  // bottom limit
  for (b = h - 1; b >= 0; b--) {
    int sum = 0;
    for (int j = 0; j < h; j++) {
      sum += edges.at<uchar>(b, j);
    }
    if (sum > 0) {
      break;
    }
  }
  return {u, b, l, r};
}

CropBounds ComputeCropBounds(const std::string& image) {
  // read an image from memory buffer and force conversion to 3 channels.
  // get crop bounds
  return CannyGetCropBounds(cv::imdecode(cv::Mat(std::vector<char>(image.begin(), image.end())), CV_LOAD_IMAGE_COLOR),
                            35, 70, 3);
}

cv::Mat CropImage(const cv::Mat& src, const CropBounds& cb) {
  if ((cb.right - cb.left) > 0 && (cb.bottom - cb.up) > 0) {
    return src(cv::Rect(cb.left, cb.up, cb.right - cb.left + 1, cb.bottom - cb.up + 1));
  } else {
    return src;
  }
}
}  // namespace similia
