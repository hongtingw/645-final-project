#ifndef LENET_INFERENCE_MNISTREADER_H
#define LENET_INFERENCE_MNISTREADER_H

#include <string>
#include <fstream>

#include "GreyscaleImage.h"

class MnistReader {
 public:
  MnistReader(const std::string& mnist_image_path, const std::string& mnist_label_path);

  ~MnistReader();

  /**
   * Read the next sample in the loaded MNIST dataset.
   * @param image Grayscale image in the sample.
   * @param label 0-9 label of the sample.
   * @return true on success; false on reaching the end of the dataset.
   */
  bool next(GreyscaleImage& image, uchar& label);
 private:
  int num_images_;
  int image_rows_;
  int image_cols_;
  int cnt_ = 0;
  std::ifstream image_fin_;
  std::ifstream label_fin_;
};

#endif //LENET_INFERENCE_MNISTREADER_H
