#ifndef LENET_INFERENCE_CONVLAYER_H
#define LENET_INFERENCE_CONVLAYER_H

#include "Layer.h"
#include "../blas/Blas.h"

class ConvLayer : public Layer {
 public:
  ConvLayer(int kernel_size, int padding, int stride);
  cv::Mat forward(const cv::Mat& input) final;
  cv::Size outputShape() const final;
 private:
  int kernel_size_;
  int padding_;
  int stride_;
};

#endif //LENET_INFERENCE_CONVLAYER_H
