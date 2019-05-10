#include <iostream>

#include "InferenceEngine.h"
#include "MnistReader.h"

int main() {
  InferenceEngine inference_engine("model/lenet_model.json", "model/lenet_weights.bin");
  MnistReader mnist_reader("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
  GreyscaleImage image{};
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