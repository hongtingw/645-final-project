#include "Blas.h"

cv::Mat Blas::multiplyOpenCV(const cv::Mat &a, const cv::Mat &b) {
  return a * b;
}

cv::Mat Blas::multiplyOptimized(const cv::Mat &a, const cv::Mat &b) {
  // TODO(Zhe Yang): Optimize 2D matrix multiplication here.
  return a * b;
}

cv::Mat Blas::multiplyNaive(const cv::Mat &a, const cv::Mat &b) {
  CV_CheckEQ(a.cols, b.rows, "Matrix dimensions do not match!");
  CV_CheckEQ(a.type(), CV_32F, "Matrix A is of unsupported type!");
  CV_CheckEQ(b.type(), CV_32F, "Matrix B is of unsupported type!");
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

cv::Mat Blas::multiply(const cv::Mat &a, const cv::Mat &b) {
  switch (mat_op_impl_) {
    case NAIVE:return multiplyNaive(a, b);
    case OPENCV:return multiplyOpenCV(a, b);
    case OPT:return multiplyOptimized(a, b);
  }
}

cv::Mat Blas::addOpenCV(const cv::Mat &a, const cv::Mat &b) {
  return a + b;
}

cv::Mat Blas::addOptimized(const cv::Mat &a, const cv::Mat &b) {
  // TODO(Zhe Yang): Optimize matrix summation here.
  return a + b;
}

cv::Mat Blas::addNaive(const cv::Mat &a, const cv::Mat &b) {
  CV_CheckEQ(a.rows, b.rows, "Matrix rows do not match!");
  CV_CheckEQ(a.cols, b.cols, "Matrix columns do not match!");
  CV_CheckEQ(a.type(), CV_32F, "Matrix A is of unsupported type!");
  CV_CheckEQ(b.type(), CV_32F, "Matrix B is of unsupported type!");
  cv::Mat c(cv::Size(a.rows, a.cols), a.type());
  for (int i = 0; i < a.rows; ++i) {
    for (int j = 0; j < b.cols; ++j) {
      c.at<float>(i, j) = a.at<float>(i, j) + b.at<float>(i, j);
    }
  }
  return c;
}

cv::Mat Blas::add(const cv::Mat &a, const cv::Mat &b) {
  switch (mat_op_impl_) {
    case NAIVE:return addNaive(a, b);
    case OPENCV:return addOpenCV(a, b);
    case OPT:return addOptimized(a, b);
  }
}