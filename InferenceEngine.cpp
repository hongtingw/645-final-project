#include "InferenceEngine.h"
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include "layers/LayerFactory.h"
#include <fstream>

using json = nlohmann::json;

InferenceEngine::InferenceEngine(const std::string &model_path,
                                 const std::string &weights_path,
                                 Blas::MatOpImpl mat_op_impl) {
  std::shared_ptr<Blas> blas = std::make_shared<Blas>(mat_op_impl);

  std::ifstream model_fin(model_path);
  CHECK(model_fin.is_open()) << "Unable to open model JSON file at " << model_path;
  json model_json;
  model_fin >> model_json;
  json layers_json = model_json["config"]["layers"];
  cv::Size x_shape;
  for (const auto &layer_json : layers_json) {
    std::unique_ptr<Layer> layer;
    const std::string layer_class_name = layer_json["class_name"].get<std::string>();
    const auto layer_config = layer_json["config"];
    if (layer_config.find("batch_input_shape") != layer_config.end()) {
      auto batch_input_shape = layer_config["batch_input_shape"];
      std::vector<int> dims;
      for (auto dim_json : batch_input_shape) {
        if (!dim_json.is_null()) {
          dims.emplace_back(dim_json.get<int>());
        }
      }
      CHECK_EQ(dims.size(), 2L) << "Currently only support 2 dimensional input!";
      x_shape = cv::Size(dims[1], dims[0]);
    }
    if (layer_class_name == "Dense") {
      // TODO(Hongting Wang): Read and fill in parameters.
      int num_output_units = layer_config["units"].get<int>();
      cv::Mat w(num_output_units, x_shape.area(), CV_32F);
      cv::Mat b(num_output_units, 1, CV_32F);
      layer = std::make_unique<DenseLayer>(w, b, blas);
    } else if (layer_class_name == "Flatten") {
      layer = std::make_unique<FlattenLayer>(x_shape);
    } else if (layer_class_name == "Dropout") {
      // Do nothing.
      continue;
    } else {
      LOG(FATAL) << layer_class_name << " layer " << layer_config["name"] << " is unsupported!";
    }

    x_shape = layer->outputShape();
    layers_.emplace_back(std::move(layer));
  }
}

uchar InferenceEngine::predict(const cv::Mat &image) {
  cv::Mat x;
  image.convertTo(x, CV_32F);
  for (auto& layer : layers_) {
    x = layer->forward(x);
  }
  CV_CheckEQ(x.rows, 10, "Network output has unexpected number of rows!");
  CV_CheckEQ(x.cols, 1, "Network output has unexpected number of columns!");
  uchar prediction = 0;
  float max_prob = x.at<float>(0);
  for (int i = 1; i < 10; ++i) {
    if (max_prob < x.at<float>(i)) {
      max_prob = x.at<float>(i);
      prediction = static_cast<uchar>(i);
    }
  }
  return prediction;
}