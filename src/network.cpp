#include "network.hpp"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/softmax.h>
#include <c10/core/ScalarType.h>
#include <spdlog/spdlog.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <utility>

#include "device.hpp"
#include "game.hpp"

namespace generals {

GeneralsNetworkImpl::GeneralsNetworkImpl() : direction_fc(nullptr) {
  auto device = get_device();
  auto input_channel = game::type_count + 2;

  conv_layers = torch::nn::Sequential(
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(input_channel, 32, 3).stride(1).padding(1)),
      torch::nn::BatchNorm2d(32), torch::nn::ReLU(), torch::nn::Dropout(0.3),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
      torch::nn::BatchNorm2d(64), torch::nn::ReLU(), torch::nn::Dropout(0.3),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
      torch::nn::BatchNorm2d(128), torch::nn::ReLU(), torch::nn::Dropout(0.3));
  conv_layers->to(device);

  fc_layers = torch::nn::Sequential(
      torch::nn::Linear(128, 64), torch::nn::ReLU(), torch::nn::Dropout(0.3),
      torch::nn::Linear(64, 32), torch::nn::ReLU(), torch::nn::Dropout(0.3));
  fc_layers->to(device);

  direction_fc = register_module("direction_linear", torch::nn::Linear(32, 4));
  direction_fc->to(device);
}

std::pair<at::Tensor, at::Tensor>
GeneralsNetworkImpl::forward(torch::Tensor x, torch::Tensor action_mask) {
  x = x.unsqueeze(0);
  action_mask = action_mask.unsqueeze(0).unsqueeze(1);

  x = conv_layers->forward(x);

  action_mask = action_mask.expand_as(x);
  x = x.masked_fill((1 - action_mask).to(torch::kBool), -1e9);

  auto from_probs = torch::softmax(x.view({x.size(0), x.size(1), -1}), 2);
  from_probs = from_probs.view({x.size(0), x.size(1), x.size(2), x.size(3)});

  auto pooled = torch::nn::functional::adaptive_max_pool2d(x, {{1, 1}});
  pooled = pooled.view({pooled.size(0), -1});

  auto features = fc_layers->forward(pooled);
  auto direction_probs = torch::softmax(direction_fc->forward(features), 1);

  return {from_probs.squeeze(0), direction_probs.squeeze(0)};
}

std::pair<game::Coord, game::Step::Direction>
select_action(torch::Tensor from_probs, torch::Tensor direction_probs) {
  auto board_probs = from_probs.sum(0);
  int m = board_probs.size(1);

  board_probs = torch::softmax(board_probs.view({-1}), 0);

  auto from_index = board_probs.multinomial(1).item<int>();

  game::Coord from = {from_index / m, from_index % m};

  auto direction_index = direction_probs.multinomial(1).item<int>();
  game::Step::Direction direction =
      static_cast<game::Step::Direction>(direction_index);

  return {from, direction};
}

} // namespace generals
