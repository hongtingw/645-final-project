/// Author: Kai Yu(cn.ken.yu@gmail.com)
/// All rights reserved.

#ifndef LENET_INFERENCE_INFERENCEPIPELINE_H
#define LENET_INFERENCE_INFERENCEPIPELINE_H

#include "InferenceEngine.h"
#include "MnistReader.h"

class InferencePipeline {
 public:
  enum PipelineType {
    SEQUENTIAL,
    BATCHED,
    BATCHED_AND_PREFETCH,
  };

  InferencePipeline(PipelineType pipeline_type, int batch_size=256);

  // Return accuracy.
  double test(InferenceEngine &engine, MnistReader &reader);

 private:
  double runSequential(InferenceEngine& engine, MnistReader& reader);
  double runBatched(InferenceEngine& engine, MnistReader& reader);
  double runBatchedAndPrefetch(InferenceEngine& engine, MnistReader& reader);

  PipelineType pipeline_type_;
  int batch_size_;
};

#endif //LENET_INFERENCE_INFERENCEPIPELINE_H
