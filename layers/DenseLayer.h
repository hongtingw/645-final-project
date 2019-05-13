#ifndef LENET_INFERENCE_DENSELAYER_H
#define LENET_INFERENCE_DENSELAYER_H

#include "Layer.h"
#include "../blas/Blas.h"

class DenseLayer : public Layer {
 public:
  enum Activation {
    RELU,
    SOFTMAX,
  };

  DenseLayer(cv::Mat w, cv::Mat b, std::shared_ptr<Blas> blas, Activation activation);
  cv::Mat forward(const cv::Mat &input) final;
  cv::Size outputShape() const final;
 private:
  cv::Mat w_;
  cv::Mat b_;
  std::shared_ptr<Blas> blas_;
  Activation activation_;
};

#endif //LENET_INFERENCE_DENSELAYER_H
