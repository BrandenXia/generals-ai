#ifndef GENERALS_NETWORK_HPP
#define GENERALS_NETWORK_HPP

#include <ATen/core/TensorBody.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <utility>

#include "game.hpp"

namespace generals {

struct GeneralsNetworkImpl : torch::nn::Module {
  torch::nn::Sequential conv_layers;
  torch::nn::Sequential fc_layers;
  torch::nn::Linear direction_fc;

  GeneralsNetworkImpl();
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x,
                                                  torch::Tensor action_mask);
};
TORCH_MODULE(GeneralsNetwork);

std::pair<game::Coord, game::Step::Direction>
select_action(torch::Tensor from_probs, torch::Tensor direction_probs);

} // namespace generals

#endif
