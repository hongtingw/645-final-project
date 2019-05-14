#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "InferenceEngine.h"
#include "MnistReader.h"
#include "InferencePipeline.h"

DEFINE_string(mat_op_impl, "naive", "Select matrix operation implementation (naive, opencv, opt)");
DEFINE_string(pipeline_type, "seq", "Select pipeline type (seq, prefetch)");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Blas::MatOpImpl mat_op_impl = Blas::MatOpImpl::NAIVE;
  if (FLAGS_mat_op_impl == "naive") {
    mat_op_impl = Blas::MatOpImpl::NAIVE;
  } else if (FLAGS_mat_op_impl == "opencv") {
    mat_op_impl = Blas::MatOpImpl::OPENCV;
  } else if (FLAGS_mat_op_impl == "opt") {
    mat_op_impl = Blas::MatOpImpl::OPT;
  } else {
    LOG(FATAL) << "Unknown matrix operation implementation name: " << FLAGS_mat_op_impl;
  }

  InferencePipeline::PipelineType pipeline_type = InferencePipeline::PipelineType::SEQUENTIAL;
  if (FLAGS_pipeline_type == "seq") {
    pipeline_type = InferencePipeline::PipelineType::SEQUENTIAL;
  } else if (FLAGS_pipeline_type == "prefetch") {
    pipeline_type = InferencePipeline::PipelineType::PREFETCH;
  } else {
    LOG(FATAL) << "Unknown pipeline type name: " << FLAGS_mat_op_impl;
  }

  InferenceEngine inference_engine("model/lenet_model.json", "model/lenet_weights.bin", mat_op_impl);
  MnistReader mnist_reader("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
  InferencePipeline pipeline(pipeline_type);

  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  // Test the inference engine on the MNIST dataset.
  const double accuracy = pipeline.test(inference_engine, mnist_reader);

  const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  const std::chrono::duration<double, std::milli> total_time = end - start;
  const double avg_inference_time = total_time.count() / mnist_reader.getNumSamples();

  LOG(INFO) << FLAGS_mat_op_impl << "+" << FLAGS_pipeline_type << "\t\t"
            << "accuracy=" << (100.f * accuracy) << "%" << "\t\t"
            << "average inference time=" << avg_inference_time << "ms";

  return 0;
}