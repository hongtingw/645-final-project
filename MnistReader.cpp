#include <iostream>
#include <opencv2/core/mat.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "MnistReader.h"

DEFINE_int32(additional_preprocessing_time, 0, "Add additional preprocessing time to test prefetching pipeline.");

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
  CHECK(image_fin_.is_open());
  CHECK(label_fin_.is_open());
  int magic_number;

  // Read the header of the image file.
  magic_number = readInt(image_fin_);
  CHECK_EQ(magic_number, 2051);
  num_samples_ = readInt(image_fin_);
  image_rows_ = readInt(image_fin_);
  image_cols_ = readInt(image_fin_);

  // Read the header of the label file.
  magic_number = readInt(label_fin_);
  CHECK_EQ(magic_number, 2049);
  int num_labels = readInt(label_fin_);
  CHECK_EQ(num_labels, num_samples_);
}

bool MnistReader::next(cv::Mat& img, uchar &label) {
  if (cnt_ < num_samples_) {
    if (FLAGS_additional_preprocessing_time) {
      usleep(FLAGS_additional_preprocessing_time);
    }
    img = cv::Mat(image_rows_, image_cols_, CV_8U);
    image_fin_.read(reinterpret_cast<char *>(img.data), image_rows_ * image_cols_ * sizeof(uchar));
    label_fin_.read(reinterpret_cast<char *>(&label), sizeof(uchar));
    ++cnt_;
    return true;
  } else {
    return false;
  }
}

MnistReader::~MnistReader() {
  image_fin_.close();
  label_fin_.close();
}
