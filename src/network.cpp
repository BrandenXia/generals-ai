#include "network.hpp"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/adaptive_avg_pool1d.h>
#include <ATen/ops/softmax.h>
#include <c10/core/ScalarType.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <utility>

#include "game.hpp"

namespace generals {

GeneralsNetwork::GeneralsNetwork(unsigned int w, unsigned int h) {
  conv_layers = torch::nn::Sequential(
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(20, 32, 3).stride(1).padding(1)),
      torch::nn::ReLU(),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
      torch::nn::ReLU(),
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
      torch::nn::ReLU());
  direction_fc = register_module("direction_fc", torch::nn::Linear(128, 4));
}

std::pair<at::Tensor, at::Tensor>
GeneralsNetwork::forward(torch::Tensor x, torch::Tensor action_mask) {
  x = conv_layers->forward(x);
  x = torch::adaptive_avg_pool2d(x, {1, 1}).view({x.size(0), -1});
  x = x + torch::nn::functional::softmax(x, 1) * x;

  auto from_digit = x;
  from_digit = from_digit.masked_fill((1 - action_mask).to(torch::kBool), -1e9);
  auto from_probs = torch::softmax(from_digit, 1);

  auto direction_probs = torch::softmax(direction_fc->forward(x), 1);

  return {from_probs, direction_probs};
}

std::pair<game::Coord, game::Step::Direction>
GeneralsNetwork::select_action(const game::PlayerBoard &board) {
  auto action_mask = board.action_mask();
  auto [from_probs, direction_probs] = forward(board.to_tensor(), action_mask);

  auto from_index = from_probs.multinomial(1).item<int>();
  int m = board.extent(1);
  game::Coord from = {from_index / m, from_index % m};

  auto direction_index = direction_probs.multinomial(1).item<int>();
  game::Step::Direction direction =
      static_cast<game::Step::Direction>(direction_index);

  return {from, direction};
}

} // namespace generals
