#include <iostream>
#include <glog/logging.h>
#include <chrono>

#include "InferenceEngine.h"
#include "MnistReader.h"
#include "InferencePipeline.h"

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

  const std::chrono::time_point start = std::chrono::high_resolution_clock::now();

  InferencePipeline pipeline(InferencePipeline::PipelineType::SEQUENTIAL);
  const double accuracy = pipeline.test(inference_engine, mnist_reader);

  const std::chrono::time_point end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> total_time = end - start;
  const double avg_inference_time = total_time.count() / mnist_reader.getNumSamples();

  LOG(INFO) << "Accuracy: " << (100.f * accuracy) << "%";
  LOG(INFO) << "Average inference time: " << avg_inference_time << "ms";

  return 0;
}