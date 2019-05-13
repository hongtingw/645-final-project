#include <utility>

#include "DenseLayer.h"

DenseLayer::DenseLayer(cv::Mat w, cv::Mat b, std::shared_ptr<Blas> blas, Activation activation)
    : w_(std::move(w)), b_(std::move(b)), blas_(std::move(blas)), activation_(activation) {
}

cv::Mat DenseLayer::forward(const cv::Mat &input) {
  auto output = blas_->add(blas_->multiply(w_, input), b_);
  switch (activation_) {
    case RELU:
      for (int r = 0; r < output.rows; ++r) {
        for (int c = 0; c < output.cols; ++c) {
          output.at<float>(r, c) = std::max(0.f, output.at<float>(r, c));
        }
      }
      break;
    case SOFTMAX:
      // Note: Currently the model inputs to Softmax is either too large or too small, and applying Softmax will cause
      // issue. For now, we do nothing, and this does not affect the prediction.
      break;
  }
  return output;
}

cv::Size DenseLayer::outputShape() const {
  return b_.size();
}
