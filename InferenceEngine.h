#ifndef LENET_INFERENCE_ESTIMATOR_H
#define LENET_INFERENCE_ESTIMATOR_H

#include <string>
#include <opencv2/opencv.hpp>
#include "blas/Blas.h"
#include "layers/Layer.h"

class InferenceEngine {
 public:
  InferenceEngine(const std::string& model_path, const std::string& weights_path, Blas::MatOpImpl mat_op_impl=Blas::NAIVE);
  uchar predict(const cv::Mat& image);
 private:
  std::vector<std::unique_ptr<Layer>> layers_;
};

#endif //LENET_INFERENCE_ESTIMATOR_H
