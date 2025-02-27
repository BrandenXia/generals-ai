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

struct ResidualBlock : torch::nn::Module {
  torch::nn::Sequential layers;

  ResidualBlock(int in_channels, int out_channels);

  torch::Tensor forward(torch::Tensor x);
};

torch::Tensor action_mask(torch::Tensor x);

struct GeneralsNetworkImpl : torch::nn::Module {
  game::Player player;
  std::pair<int, int> max_size;

  torch::nn::Sequential input_layers;

  torch::nn::Sequential resnet_block;

  torch::nn::Sequential policy_conv;
  torch::nn::Linear policy_fc;

  torch::nn::Sequential value_conv;
  torch::nn::Sequential value_fc;

  GeneralsNetworkImpl();
  GeneralsNetworkImpl(game::Player player, std::pair<int, int> max_size);

  torch::Tensor encode(const PlayerBoard &board, unsigned int tick,
                       game::Coord general) const;

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

  // convert policy index to step
  game::Step idx2step(unsigned int idx) const;
  game::Step select_action(torch::Tensor policy) const;
};
TORCH_MODULE(GeneralsNetwork);

void create(game::Player player, std::pair<int, int> max_size,
            const std::filesystem::path &network_path);
void save(GeneralsNetwork &network, const std::filesystem::path &network_path);
GeneralsNetwork load(const std::filesystem::path &network_path);

std::string info(const std::filesystem::path &network_path);

} // namespace generals::network

namespace generals {

using network::GeneralsNetwork;

}

#endif
