#include <utility>
#include <thread>
#include <mutex>

#include <glog/logging.h>

#include "InferencePipeline.h"

InferencePipeline::InferencePipeline(InferencePipeline::PipelineType pipeline_type) :
    pipeline_type_(pipeline_type) {

}

double InferencePipeline::test(InferenceEngine &engine, MnistReader &reader) {
  switch (pipeline_type_) {
    case SEQUENTIAL:return runSequential(engine, reader);
    case PREFETCH:return runPrefetch(engine, reader);
    default:return 0;
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

class PrefetchReader {
 public:
  typedef std::pair<cv::Mat, uchar> Sample;
  inline explicit PrefetchReader(MnistReader &reader, int buffer_size = 10000) : reader_(reader) {
    prefetch_thread_ = std::thread([this, buffer_size]() {
      cv::Mat image;
      uchar label;
      while (true) {
        if (buffer_.size() < buffer_size) {
          if (!reader_.next(image, label)) {
            ended_ = true;
            break;
          }
          lock_.lock();
          buffer_.emplace(std::make_pair<>(image, label));
          lock_.unlock();
        } else {
          usleep(1);
        }
      }
    });
  }
  inline bool next(cv::Mat &img, uchar &label) {
    while (buffer_.empty() && !ended_) {
      usleep(1);
    }
    if (buffer_.empty()) {
      return false;
    }
    lock_.lock();
    img = buffer_.front().first;
    label = buffer_.front().second;
    buffer_.pop();
    lock_.unlock();
    return true;
  }
  inline ~PrefetchReader() {
    prefetch_thread_.join();
  }
 private:
  std::queue<Sample> buffer_;
  MnistReader &reader_;
  std::thread prefetch_thread_;
  bool ended_ = false;
  std::mutex lock_;
};

double InferencePipeline::runPrefetch(InferenceEngine &engine, MnistReader &reader) {
  cv::Mat image;
  uchar label;
  int num_correct = 0;
  PrefetchReader prefetch_reader(reader);
  while (prefetch_reader.next(image, label)) {
    uchar pred = engine.predict(image);
    if (pred == label) {
      ++num_correct;
    }
  }
  return static_cast<double>(num_correct) / reader.getNumSamples();
}

