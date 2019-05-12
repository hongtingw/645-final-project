#include <iostream>
#include <glog/logging.h>

#include "InferenceEngine.h"
#include "MnistReader.h"

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  Blas::MatOpImpl mat_op_impl = Blas::MatOpImpl::NAIVE;
  if (argc > 1) {
    char* mat_mul_impl_name = argv[1];
    for (int i = 0; i < strlen(mat_mul_impl_name); ++i) {
      mat_mul_impl_name[i] = static_cast<char>(tolower(mat_mul_impl_name[i]));
    }
    if (!strcmp(mat_mul_impl_name, "naive")) {
      mat_op_impl = Blas::MatOpImpl::NAIVE;
    } else if (!strcmp(mat_mul_impl_name, "opencv")) {
      mat_op_impl = Blas::MatOpImpl::OPENCV;
    } else if (!strcmp(mat_mul_impl_name, "opt") || !strcmp(mat_mul_impl_name, "optimized")) {
      mat_op_impl = Blas::MatOpImpl::OPT;
    } else {
      std::cerr << "Unknown matrix operation implementation name: " << mat_mul_impl_name << std::endl;
      return -1;
    }
  }
  InferenceEngine inference_engine("model/lenet_model.json", "model/lenet_weights.bin", mat_op_impl);
  MnistReader mnist_reader("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
  cv::Mat image;
  uchar label;
  int num_samples = 0;
  int num_correct = 0;
  while (mnist_reader.next(image, label)) {
    ++num_samples;
    uchar pred = inference_engine.predict(image);
    if (pred == label) {
      ++num_correct;
    } else {
      std::cout << "Misclassified " << int(label) << " into " << int(pred) << "!" << std::endl;
    }
  }
  std::cout << "Accuracy: " << (100.f * num_correct / num_samples) << "%" << std::endl;
  return 0;
}