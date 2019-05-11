#ifndef LENET_INFERENCE_ESTIMATOR_H
#define LENET_INFERENCE_ESTIMATOR_H

#include <string>
#include <opencv2/opencv.hpp>

class InferenceEngine {
 public:
  InferenceEngine(const std::string& model_path, const std::string& weights_path);
  uchar predict(const cv::Mat& image);
 private:
  /**
   * Multiply 2 matrices.
   * @param a The first matrix.
   * @param b The second matrix.
   * @return Product of the 2 matrices.
   */
  template <typename DTYPE>
  cv::Mat multiply(const cv::Mat& a, const cv::Mat& b) {
    // TODO(Zhe Yang): Optimize 2D matrix multiplication here.
    assert(a.cols == b.rows);
    cv::Mat c(a.rows, b.cols, a.type());
    DTYPE* p = c.data;
    for (int i = 0; i < a.rows; ++i) {
      for (int j = 0; j < b.cols; ++j) {
        for (int k = 0; k < a.cols; ++k) {
          (*p++) = a.at<DTYPE>(i, j) * b.at<DTYPE>(j, k);
        }
      }
    }
    return c;
  }
};

#endif //LENET_INFERENCE_ESTIMATOR_H
