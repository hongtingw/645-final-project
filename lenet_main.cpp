#include <iostream>

#include "InferenceEngine.h"
#include "MnistReader.h"

int main() {
  InferenceEngine inference_engine("model/lenet_model.json", "model/lenet_weights.bin");
  MnistReader mnist_reader("t10k-images-idx3-ubyte", "t10k-labels-idx3-ubyte");
  MnistImage image{};
  int label;
  int num_samples = 0;
  int num_correct = 0;
  while (mnist_reader.next(image, label)) {
    ++num_samples;
    int pred = inference_engine.predict(image);
    if (pred == label) {
      ++num_correct;
    }
  }
  std::cout << "Accuracy: " << (100.f * num_correct / num_samples) << "%" << std::endl;
  return 0;
}