/// Author: Kai Yu(cn.ken.yu@gmail.com)
/// All rights reserved.

#include "InferencePipeline.h"

InferencePipeline::InferencePipeline(InferencePipeline::PipelineType pipeline_type) :
    pipeline_type_(pipeline_type) {

}

double InferencePipeline::test(InferenceEngine &engine, MnistReader &reader) {
  switch (pipeline_type_) {
    case SEQUENTIAL:
      return runSequential(engine, reader);
    case PREFETCH:
      return runPrefetch(engine, reader);
    default:
      return 0;
  }
}

double InferencePipeline::runSequential(InferenceEngine &engine, MnistReader &reader) {
  cv::Mat image;
  uchar label;
  int num_correct = 0;
  while (reader.next(image, label)) {
    uchar pred = engine.predict(image);
    if (pred == label) {
      ++num_correct;
    }
  }
  return static_cast<double>(num_correct) / reader.getNumSamples();
}

double InferencePipeline::runPrefetch(InferenceEngine &engine, MnistReader &reader) {
  // TODO(Zhe YANG): Implement BatchedAndPrefetch pipeline. Batches are prepared in a separate thread from the network
  // processing.
  return runSequential(engine, reader);
}

