#ifndef LENET_INFERENCE_MNISTDATA_H
#define LENET_INFERENCE_MNISTDATA_H

#include <vector>

typedef unsigned char uchar;

struct MnistImage {
  int height;
  int width;
  std::vector<uchar> grayscale;
};

#endif //LENET_INFERENCE_MNISTDATA_H
