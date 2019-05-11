#include "InferenceEngine.h"

InferenceEngine::InferenceEngine(const std::string &model_path,
                                 const std::string &weights_path,
                                 MatMulImpl mat_mul_impl)
    : mat_mul_impl_(mat_mul_impl) {
  // TODO
}

uchar InferenceEngine::predict(const cv::Mat &image) {
  // TODO
  return 0;
}

cv::Mat InferenceEngine::multiplyOpenCV(const cv::Mat &a, const cv::Mat &b) {
  return a * b;
}

cv::Mat InferenceEngine::multiplyOptimized(const cv::Mat &a, const cv::Mat &b) {
  // TODO(Zhe Yang): Optimize 2D matrix multiplication here.
  return a * b;
}

cv::Mat InferenceEngine::multiplyNaive(const cv::Mat &a, const cv::Mat &b) {
  CV_CheckEQ(a.cols, b.rows, "Matrix dimensions do not match!");
  CV_CheckEQ(a.type(), CV_32F, "Matrix A is of wrong type!");
  CV_CheckEQ(b.type(), CV_32F, "Matrix B is of wrong type!");
  cv::Mat c = cv::Mat::zeros(cv::Size(a.rows, b.cols), a.type());
  for (int i = 0; i < a.rows; ++i) {
    for (int j = 0; j < b.cols; ++j) {
      for (int k = 0; k < a.cols; ++k) {
        c.at<float>(i, j) += a.at<float>(i, j) * b.at<float>(j, k);
      }
    }
  }
  return c;
}

cv::Mat InferenceEngine::multiply(const cv::Mat &a, const cv::Mat &b) {
  switch (mat_mul_impl_) {
    case NAIVE:return multiplyNaive(a, b);
    case OPENCV:return multiplyOpenCV(a, b);
    case OPT:return multiplyOptimized(a, b);
  }
}