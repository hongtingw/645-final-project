#ifndef LENET_INFERENCE_LAYER_H
#define LENET_INFERENCE_LAYER_H

#include <opencv2/opencv.hpp>

class Layer {
 public:
  virtual cv::Mat forward(const cv::Mat& input) = 0;
  virtual cv::Size outputShape() const = 0;
  virtual ~Layer() = default;
};

#endif //LENET_INFERENCE_LAYER_H
