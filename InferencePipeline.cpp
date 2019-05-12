/// Author: Kai Yu(cn.ken.yu@gmail.com)
/// All rights reserved.

#include "InferencePipeline.h"

InferencePipeline::InferencePipeline(InferencePipeline::PipelineType pipeline_type, int batch_size) :
    pipeline_type_(pipeline_type), batch_size_(batch_size) {

}

double InferencePipeline::test(InferenceEngine &engine, MnistReader &reader) {
  switch (pipeline_type_) {
    case SEQUENTIAL:
      return runSequential(engine, reader);
    case BATCHED:
      return runBatched(engine, reader);
    case BATCHED_AND_PREFETCH:
      return runBatchedAndPrefetch(engine, reader);
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

double InferencePipeline::runBatched(InferenceEngine &engine, MnistReader &reader) {
  // TODO(Zhe YANG): Implement BatchedAndPrefetch pipeline. Specifically, collect inputs into batches, and feed the
  // data in batches to the network. Need to implement batch processing in the layers.
  return runSequential(engine, reader);
}

double InferencePipeline::runBatchedAndPrefetch(InferenceEngine &engine, MnistReader &reader) {
  // TODO(Zhe YANG): Implement BatchedAndPrefetch pipeline. Batches are prepared in a separate thread from the network
  // processing.
  return runSequential(engine, reader);
}

