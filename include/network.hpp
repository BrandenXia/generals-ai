#ifndef GENERALS_NETWORK_HPP
#define GENERALS_NETWORK_HPP

#include <ATen/core/TensorBody.h>
#include <filesystem>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <utility>

#include "game.hpp"

namespace generals::network {

struct GeneralsNetworkImpl : torch::nn::Module {
  game::Player player;
  std::pair<int, int> max_size;

  torch::nn::Sequential conv_layers;
  torch::nn::Sequential residual_block;
  torch::nn::Sequential from_fc;
  torch::nn::Sequential direction_fc;

  GeneralsNetworkImpl();
  GeneralsNetworkImpl(game::Player player, std::pair<int, int> max_size);

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x,
                                                  torch::Tensor action_mask);
};
TORCH_MODULE(GeneralsNetwork);

void create(game::Player player, std::pair<int, int> max_size,
            const std::filesystem::path &network_path);
void save(GeneralsNetwork &network, const std::filesystem::path &network_path);
GeneralsNetwork load(const std::filesystem::path &network_path);

torch::Tensor encode(const PlayerBoard &board, unsigned int tick,
                     game::Coord general);

std::pair<game::Coord, game::Step::Direction>
select_action(torch::Tensor from_probs, torch::Tensor direction_probs);

std::string info(const std::filesystem::path &network_path);

} // namespace generals::network

namespace generals {

using network::GeneralsNetwork;

}

#endif
