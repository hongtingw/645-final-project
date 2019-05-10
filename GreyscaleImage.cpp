#include "GreyscaleImage.h"

GreyscaleImage::GreyscaleImage(int rows, int cols) : Tensor(std::array<int, 2>{rows, cols}) {
}

GreyscaleImage::GreyscaleImage() {

}

void GreyscaleImage::fromBytes(std::istream &byte_stream) {
  byte_stream.read(reinterpret_cast<char *>(data_), rows() * cols() * sizeof(uchar));
}