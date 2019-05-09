#ifndef LENET_INFERENCE_MNISTREADER_H
#define LENET_INFERENCE_MNISTREADER_H

#include <string>

#include "MnistImage.h"

class MnistReader {
 public:
  MnistReader(const std::string& mnist_image_path, const std::string& mnist_label_path);
  /**
   * Read the next sample in the loaded MNIST dataset.
   * @param image Grayscale image in the sample.
   * @param label 0-9 label of the sample.
   * @return true on success; false on reaching the end of the dataset.
   */
  bool next(MnistImage& image, int& label);
};

#endif //LENET_INFERENCE_MNISTREADER_H
