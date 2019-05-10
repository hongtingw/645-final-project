#ifndef LENET_INFERENCE_MNISTDATA_H
#define LENET_INFERENCE_MNISTDATA_H

#include "Tensor.h"
#include <istream>

typedef unsigned char uchar;

class GreyscaleImage : public Tensor<uchar, 2> {
 public:
  GreyscaleImage();
  GreyscaleImage(int rows, int cols);
  inline int rows() const { return dims_[0]; }
  inline int cols() const { return dims_[1]; }
  void fromBytes(std::istream& byte_stream);
};

#endif //LENET_INFERENCE_MNISTDATA_H
