#include "network.hpp"

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/softmax.h>
#include <algorithm>
#include <c10/core/ScalarType.h>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <torch/nn/functional/normalization.h>
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

#include "device.hpp"
#include "game.hpp"

namespace generals::network {

ResidualBlock::ResidualBlock(int in_channels, int out_channels)
    : layers(torch::nn::Conv2d(
                 torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                     .padding(1)),
             torch::nn::BatchNorm2d(out_channels), torch::nn::ReLU(),
             torch::nn::Conv2d(
                 torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                     .padding(1)),
             torch::nn::BatchNorm2d(out_channels))

{
  register_module("layers", layers);
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
  return torch::relu(layers->forward(x) + x);
}

inline constexpr auto META_PLAYER_KEY = "player";
inline constexpr auto META_MAX_SIZE_KEY = "max_size";

inline constexpr unsigned int num_channels = 64;
inline constexpr unsigned int num_resnet_blocks = 10;

GeneralsNetworkImpl::GeneralsNetworkImpl()
    : GeneralsNetworkImpl(std::nullopt, {0, 0}) {}

GeneralsNetworkImpl::GeneralsNetworkImpl(game::Player p, std::pair<int, int> s)
    : player(p), max_size(s),

      input_layers(
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(13, num_channels, 3).padding(1)),
          torch::nn::BatchNorm2d(num_channels), torch::nn::ReLU()),

      policy_conv(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 2, 1)),
          torch::nn::BatchNorm2d(2), torch::nn::ReLU()),
      policy_fc(2 * s.first * s.second, 4 * s.first * s.second),

      value_conv(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channels, 1, 1)),
          torch::nn::BatchNorm2d(1), torch::nn::ReLU()),
      value_fc(torch::nn::Linear(s.first * s.second, num_channels),
               torch::nn::ReLU(), torch::nn::Linear(num_channels, 1),
               torch::nn::Tanh())

{
  for (unsigned int i = 0; i < num_resnet_blocks; ++i)
    resnet_block->push_back(ResidualBlock(num_channels, num_channels));

  register_module("input_layers", input_layers);
  register_module("resnet_block", resnet_block);
  register_module("policy_conv", policy_conv);
  register_module("policy_fc", policy_fc);
  register_module("value_conv", value_conv);
  register_module("value_fc", value_fc);
}

constexpr unsigned int MAX_TICK = 500;

// channels:
// 1. self army size, normalized using z-score
// 2. opponent army size, normalized using z-score
// 3. neutral army size, normalized using z-score
// 4. self territory
// 5. enemy territory
// 6. neutral territory
// 7. obstacle
// 8. self general
// 9. enemy general if visible
// 10. fog of war, 1 if visible, 0 otherwise
// 11. tick, normalized by max tick
// 12. all 1s
// 13. all 0s
torch::Tensor GeneralsNetworkImpl::encode(const PlayerBoard &board,
                                          unsigned int tick,
                                          game::Coord general) const {
  unsigned int h = board.extent(0);
  unsigned int w = board.extent(1);
  auto player = board.player;

  auto x = at::zeros({13, h, w}).to(get_device());

  x[7][general.first][general.second] = 1;
  for (std::size_t i = 0; i < h; ++i)
    for (std::size_t j = 0; j < w; ++j) {
      const auto &tile = board[i, j];
      auto owner = tile.owner;
      auto has_owner = tile.owner.has_value();
      auto owned = tile.owner == player;
      x[0][i][j] = owned ? tile.army : 0;
      x[1][i][j] = has_owner && !owned ? tile.army : 0;
      x[2][i][j] = !has_owner ? tile.army : 0;
      x[3][i][j] = owned ? 1 : 0;
      x[4][i][j] = has_owner && !owned ? 1 : 0;
      x[5][i][j] = !has_owner ? 1 : 0;
      x[6][i][j] =
          tile.type == game::Type::Mountain ||
                  tile.type == game::Type::UnknownObstacles ||
                  (has_owner && !owned && tile.type == game::Type::City)
              ? 1
              : 0;
      x[8][i][j] =
          has_owner && !owned && tile.type == game::Type::General ? 1 : 0;
      x[9][i][j] = tile.type == game::Type::Unknown ||
                           tile.type == game::Type::UnknownObstacles
                       ? 1
                       : 0;
    }
  x[10] = std::max(tick, MAX_TICK) / static_cast<float>(MAX_TICK);
  x[11] = 1;
  x[12] = 0;

  x[0] = torch::nn::functional::normalize(x[0][0]);
  x[1] = torch::nn::functional::normalize(x[0][1]);
  x[2] = torch::nn::functional::normalize(x[0][2]);

  // pad the tensor to top left
  return torch::nn::functional::pad(
      x, torch::nn::functional::PadFuncOptions(
             {0, max_size.second - w, 0, max_size.first - h})
             .mode(torch::kConstant));
}

inline torch::Tensor action_mask(torch::Tensor x) {
  auto f = x[0][3].flatten();
  return torch::cat({f, f, f, f});
}

std::pair<at::Tensor, at::Tensor>
GeneralsNetworkImpl::forward(torch::Tensor x) {
  x = input_layers->forward(x);
  x = resnet_block->forward(x);

  auto policy = policy_conv->forward(x);
  policy = policy.view({-1, 2 * max_size.first * max_size.second});
  policy = policy_fc->forward(policy);
  policy = policy.masked_fill(action_mask(x) == 0, -1e9);
  policy = torch::softmax(policy, 1);

  auto value = value_conv->forward(x);
  value = value.view({-1, max_size.first * max_size.second});
  value = value_fc->forward(value);

  return {policy, value};
}

game::Step GeneralsNetworkImpl::select_action(torch::Tensor policy) const {
  auto idx = torch::argmax(policy).item<int>();
  unsigned int board_size = max_size.first * max_size.second;

  // major tile, policy interpret as [t1 up, t2 up, ..., t1 left, t2 left, ...]
  auto direction = idx / board_size;
  auto coord = idx % board_size;

  game::Coord pos{coord / max_size.second, coord % max_size.second};

  return {player, pos, static_cast<game::Step::Direction>(direction)};
}

void create(game::Player player, std::pair<int, int> max_size,
            const std::filesystem::path &network_path) {
  network::GeneralsNetwork network{player, max_size};
  save(network, network_path);
}

void save(GeneralsNetwork &network, const std::filesystem::path &network_path) {
  std::filesystem::create_directories(network_path.parent_path());

  // meta in json
  std::ofstream meta_file{
      std::filesystem::path{network_path}.replace_extension("json")};
  nlohmann::json meta = {{META_PLAYER_KEY, network->player.value()},
                         {META_MAX_SIZE_KEY, network->max_size}};
  meta_file << meta;

  // network
  torch::save(network, network_path);
}

GeneralsNetwork load(const std::filesystem::path &network_path) {
  std::ifstream meta_file{
      std::filesystem::path{network_path}.replace_extension("json")};
  nlohmann::json meta = nlohmann::json::parse(meta_file);

  GeneralsNetwork network{
      game::Player{meta[META_PLAYER_KEY].template get<int>()},
      meta[META_MAX_SIZE_KEY].template get<std::pair<int, int>>()};

  torch::load(network, network_path);

  return network;
}

std::string info(const std::filesystem::path &network_path) {
  GeneralsNetwork network = load(network_path);
  return std::format(
      "Network Name: {}\nParameter size: {}\nPlayer: {}\nMax size: {}x{}",
      network_path.filename().replace_extension().string(),
      network->parameters().size(), network->player, network->max_size.first,
      network->max_size.second);
}

} // namespace generals::network
