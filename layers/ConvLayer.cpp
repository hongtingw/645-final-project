#include "ConvLayer.h"

ConvLayer::ConvLayer(int kernel_size, int padding, int stride)
    : kernel_size_(kernel_size), padding_(padding), stride_(stride) {

}

cv::Mat ConvLayer::forward(const cv::Mat &input) {
  // TODO(Hongting Wang): Implement convolution layer.
  return cv::Mat();
}

cv::Size ConvLayer::outputShape() const {
  // TODO(Hongting Wang): Implement convolution layer output shape.
  return cv::Size();
}
