#ifndef LENET_INFERENCE_BLAS_H
#define LENET_INFERENCE_BLAS_H

#include <opencv2/opencv.hpp>

class Blas {
 public:
  /**
   * Enumeration of different matrix operations available.
   */
  enum MatOpImpl {
    NAIVE,
    OPENCV,
    OPT,
  };

  inline Blas(MatOpImpl mat_mul_impl) : mat_op_impl_(mat_mul_impl) {}

  /**
   * Multiplication of 2 given matrices. Implementation is specified on construction.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return product of the 2 matrices.
   */
  cv::Mat multiply(const cv::Mat& a, const cv::Mat& b);

  /**
   * Summation of 2 given matrices. Implementation is specified on construction.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return summation of the 2 matrices.
   */
  cv::Mat add(const cv::Mat& a, const cv::Mat& b);

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
   * OpenCV implementation for summing 2 matrices with float type.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return summation of the 2 matrices.
   */
  cv::Mat addOpenCV(const cv::Mat& a, const cv::Mat& b);

  /**
   * Optimized implementation for summing 2 matrices with float type.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return summation of the 2 matrices.
   */
  cv::Mat addOptimized(const cv::Mat& a, const cv::Mat& b);

  /**
   * Most naive implementation for summing 2 matrices with float type.
   * @param a the first matrix.
   * @param b the second matrix.
   * @return summation of the 2 matrices.
   */
  cv::Mat addNaive(const cv::Mat& a, const cv::Mat& b);

  MatOpImpl mat_op_impl_;
};

#endif //LENET_INFERENCE_BLAS_H
