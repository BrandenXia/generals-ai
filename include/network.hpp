#ifndef GENERALS_NETWORK_HPP
#define GENERALS_NETWORK_HPP

#include <ATen/core/TensorBody.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <utility>

#include "game.hpp"

namespace generals {

struct GeneralsNetwork : torch::nn::Module {
  torch::nn::Sequential conv_layers;
  torch::nn::Linear direction_fc;

  GeneralsNetwork(unsigned int w, unsigned int h);
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x,
                                                  torch::Tensor action_mask);
  std::pair<game::Coord, game::Step::Direction>
  select_action(const game::PlayerBoard &board);
};

} // namespace generals

#endif
