#include "FlattenLayer.h"

FlattenLayer::FlattenLayer(const cv::Size &input_size) : input_size_(input_size) {

}

cv::Mat FlattenLayer::forward(const cv::Mat &input) {
  return input.reshape(1, input.channels() * input.rows * input.cols);
}

cv::Size FlattenLayer::outputShape() const {
  return cv::Size(1, input_size_.area());
}
