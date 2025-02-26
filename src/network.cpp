#include "network.hpp"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/softmax.h>
#include <c10/core/ScalarType.h>
#include <format>
#include <spdlog/spdlog.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/pooling.h>
#include <torch/serialize.h>
#include <torch/serialize/input-archive.h>
#include <utility>

#include "game.hpp"

namespace generals::network {

inline constexpr auto ARCHIVE_PLAYER_KEY = "player";
inline constexpr auto ARCHIVE_MAX_WIDTH_KEY = "max_width";
inline constexpr auto ARCHIVE_MAX_HEIGHT_KEY = "max_height";

void GeneralsNetworkImpl::save(torch::serialize::OutputArchive &archive) const {
  torch::nn::Module::save(archive);
  archive.write(ARCHIVE_PLAYER_KEY, player.value());
  archive.write(ARCHIVE_MAX_WIDTH_KEY, max_size.first);
  archive.write(ARCHIVE_MAX_HEIGHT_KEY, max_size.second);
}

void GeneralsNetworkImpl::load(torch::serialize::InputArchive &archive) {
  torch::nn::Module::load(archive);

  c10::IValue player_val;
  archive.read(ARCHIVE_PLAYER_KEY, player_val);
  player = player_val.toInt();

  c10::IValue max_width_val, max_height_val;
  archive.read(ARCHIVE_MAX_WIDTH_KEY, max_width_val);
  archive.read(ARCHIVE_MAX_HEIGHT_KEY, max_height_val);
  max_size = {max_width_val.toInt(), max_height_val.toInt()};
}

GeneralsNetworkImpl::GeneralsNetworkImpl()
    : GeneralsNetworkImpl(std::nullopt, {0, 0}) {}

GeneralsNetworkImpl::GeneralsNetworkImpl(game::Player p, std::pair<int, int> s)
    : player(p), max_size(s),
      conv_layers(torch::nn::Sequential(
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(game::type_count + 2, 32, 3).padding(1)),
          torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
          torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
          torch::nn::BatchNorm2d(128), torch::nn::ReLU(),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)),
          torch::nn::BatchNorm2d(256), torch::nn::ReLU(),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
          torch::nn::BatchNorm2d(256), torch::nn::ReLU(),
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(256, 256, 3).padding(2).dilation(2)),
          torch::nn::BatchNorm2d(256), torch::nn::ReLU())),

      residual_block(torch::nn::Sequential(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
          torch::nn::BatchNorm2d(256), torch::nn::ReLU(),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
          torch::nn::BatchNorm2d(256))),

      from_fc(torch::nn::Sequential(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 1)),
          torch::nn::Softmax(torch::nn::SoftmaxOptions(1)))),

      direction_fc(torch::nn::Sequential(
          torch::nn::Linear(256, 128), torch::nn::ReLU(),
          torch::nn::Linear(128, 4),
          torch::nn::Softmax(torch::nn::SoftmaxOptions(0)))) {

  register_module("conv_layers", conv_layers);
  register_module("residual_block", residual_block);
  register_module("from_fc", from_fc);
  register_module("direction_fc", direction_fc);
}

std::pair<at::Tensor, at::Tensor>
GeneralsNetworkImpl::forward(torch::Tensor x, torch::Tensor action_mask) {
  x = x.unsqueeze(0);
  action_mask = action_mask.unsqueeze(0);

  x = conv_layers->forward(x);
  auto residual = x;
  x = residual_block->forward(x) + residual;

  auto from_probs = from_fc->forward(x);
  from_probs = from_probs * action_mask;

  auto gap_features = torch::avg_pool2d(x, {x.size(2), x.size(3)}).squeeze();
  auto direction_probs = direction_fc->forward(gap_features);

  return {from_probs, direction_probs};
}

std::pair<game::Coord, game::Step::Direction>
select_action(torch::Tensor from_probs, torch::Tensor direction_probs) {
  auto flattened_from_idx = from_probs.argmax();
  auto idx = flattened_from_idx.item<int>();
  unsigned int m = from_probs.size(3);
  game::Coord from{idx / m, idx % m};

  int direction = direction_probs.argmax().item<int>();

  return {from, static_cast<game::Step::Direction>(direction)};
}

std::string info(const std::filesystem::path &network_path) {
  GeneralsNetwork network;
  torch::load(network, network_path);
  return std::format(
      "Network Name: {}\nParameter size: {}\nPlayer: {}\nMax size: {}x{}",
      network_path.filename().replace_extension().string(),
      network->parameters().size(), network->player, network->max_size.first,
      network->max_size.second);
}

} // namespace generals::network
