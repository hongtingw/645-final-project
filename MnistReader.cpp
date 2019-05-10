#include <iostream>
#include "MnistReader.h"

inline int readInt(std::istream& in) {
  char buffer[4];
  in.read(buffer, sizeof(int));
  int n = 0;
  for (char b : buffer) {
    n = (n << 8) | b;
  }
  return n;
}

MnistReader::MnistReader(const std::string &mnist_image_path, const std::string &mnist_label_path) :
    image_fin_(mnist_image_path, std::ios::binary),
    label_fin_(mnist_label_path, std::ios::binary) {
  assert(image_fin_.is_open());
  assert(label_fin_.is_open());
  int magic_number;

  // Read the header of the image file.
  magic_number = readInt(image_fin_);
  assert(magic_number == 2051);
  num_images_ = readInt(image_fin_);
  image_rows_ = readInt(image_fin_);
  image_cols_ = readInt(image_fin_);

  // Read the header of the label file.
  magic_number = readInt(label_fin_);
  assert(magic_number == 2049);
  int num_labels = readInt(label_fin_);
  assert(num_labels == num_images_);
}

bool MnistReader::next(GreyscaleImage &image, uchar &label) {
  if (cnt_ < num_images_) {
    image.resize(std::array<int, 2>{image_rows_, image_cols_});
    image_fin_.read(reinterpret_cast<char *>(image.data()), image_rows_ * image_cols_ * sizeof(uchar));
    label_fin_.read(reinterpret_cast<char *>(&label), sizeof(uchar));
    ++cnt_;
  }
}

MnistReader::~MnistReader() {
  image_fin_.close();
  label_fin_.close();
}
