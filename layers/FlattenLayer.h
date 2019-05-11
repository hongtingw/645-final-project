#ifndef LENET_INFERENCE_FLATTENLAYER_H
#define LENET_INFERENCE_FLATTENLAYER_H

#include "Layer.h"

class FlattenLayer : public Layer {
 public:
  explicit FlattenLayer(const cv::Size& input_size);
  cv::Mat forward(const cv::Mat& input) final;
  cv::Size outputShape() const final;
 private:
  cv::Size input_size_;
};

#endif //LENET_INFERENCE_FLATTENLAYER_H
