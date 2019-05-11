#ifndef LENET_INFERENCE_ESTIMATOR_H
#define LENET_INFERENCE_ESTIMATOR_H

#include <string>
#include <opencv2/opencv.hpp>


class InferenceEngine {
 public:
  enum MatMulImpl {
    NAIVE,
    OPENCV,
    OPT,
  };

  InferenceEngine(const std::string& model_path, const std::string& weights_path, MatMulImpl mat_mul_impl=NAIVE);
  uchar predict(const cv::Mat& image);
 private:
  /**
   * OpenCV implementation for multiplying 2 matrices with float type.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return product of the 2 matrices.
   */
  cv::Mat multiplyOpenCV(const cv::Mat& a, const cv::Mat& b);

  /**
   * Optimized implementation for multiplying 2 matrices with float type.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return product of the 2 matrices.
   */
  cv::Mat multiplyOptimized(const cv::Mat& a, const cv::Mat& b);

  /**
   * Most naive implementation for multiplying 2 matrices with float type.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return product of the 2 matrices.
   */
  cv::Mat multiplyNaive(const cv::Mat& a, const cv::Mat& b);

  /**
   * Multiplication of 2 given matrices. Implementation is specified on construction.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return product of the 2 matrices.
   */
  cv::Mat multiply(const cv::Mat& a, const cv::Mat& b);

  MatMulImpl mat_mul_impl_;
};

#endif //LENET_INFERENCE_ESTIMATOR_H
