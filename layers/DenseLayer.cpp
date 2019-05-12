#include "DenseLayer.h"

DenseLayer::DenseLayer(const cv::Mat &w, const cv::Mat &b, const std::shared_ptr<Blas> blas)
    : w_(w), b_(b), blas_(blas) {
}

cv::Mat DenseLayer::forward(const cv::Mat &input) {
  return blas_->add(blas_->multiply(w_, input), b_);
}

cv::Size DenseLayer::outputShape() const {
  return b_.size();
}
