#ifndef LENET_INFERENCE_ESTIMATOR_H
#define LENET_INFERENCE_ESTIMATOR_H

#include <string>
#include "MnistImage.h"

class InferenceEngine {
 public:
  InferenceEngine(const std::string& model_path, const std::string& weights_path);
  int predict(const MnistImage& image);
 private:
  /**
   * @tparam N_DIM Number of dimension of the tensor.
   */
  template <int N_DIM>
  class Tensor {
   private:
    std::vector<float> data_;
    std::array<int, N_DIM> dims_;
  };
  /**
   * Multiply 2 matrices.
   * @param a The first matrix.
   * @param b The second matrix.
   * @return Product of the 2 matrices.
   */
  Tensor<2> multiply(const Tensor<2>& a, const Tensor<2>& b);
};

#endif //LENET_INFERENCE_ESTIMATOR_H
