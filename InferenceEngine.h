#ifndef LENET_INFERENCE_ESTIMATOR_H
#define LENET_INFERENCE_ESTIMATOR_H

#include <string>
#include "GreyscaleImage.h"
#include "Tensor.h"

class InferenceEngine {
 public:
  InferenceEngine(const std::string& model_path, const std::string& weights_path);
  uchar predict(const GreyscaleImage& image);
 private:
  /**
   * Multiply 2 matrices.
   * @param a The first matrix.
   * @param b The second matrix.
   * @return Product of the 2 matrices.
   */
  template <typename DTYPE>
  Tensor<DTYPE, 2> multiply(const Tensor<DTYPE, 2>& a, const Tensor<DTYPE, 2>& b) {
    // TODO(Zhe Yang): Implement 2D matrix multiplication here.
  }
};

#endif //LENET_INFERENCE_ESTIMATOR_H
