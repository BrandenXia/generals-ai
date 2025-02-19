#include "train.hpp"

#include <random>
#include <spdlog/fmt/std.h>
#include <spdlog/spdlog.h>
#include <torch/optim/adam.h>
#include <torch/serialize.h>

#include "device.hpp"
#include "dirs.hpp"
#include "evaluation.hpp"
#include "game.hpp"
#include "interaction.hpp"
#include "network.hpp"

namespace generals::train {

inline const auto NETWORK_PATH = DATA_DIR / "network.pt";

std::deque<double> reward_history;
const int history_size = 100; // Number of episodes to average over

double calculate_baseline() {
  if (reward_history.empty()) return 0.0;
  double sum =
      std::accumulate(reward_history.begin(), reward_history.end(), 0.0);
  return sum / reward_history.size();
}

eval::AlgoEval eval;
at::Tensor loss_calculate(game::Player player, game::Game game,
                          at::Tensor from_probs, at::Tensor direction_probs,
                          game::Coord from, game::Step::Direction direction) {
  auto device = get_device();

  double reward = eval(game, player);

  reward_history.push_back(reward);
  if (reward_history.size() > history_size) { reward_history.pop_front(); }
  auto reward_tensor =
      torch::tensor({reward}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device);

  double baseline = calculate_baseline();
  auto baseline_tensor =
      torch::tensor({baseline}, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device);

  auto advantage = reward_tensor - baseline_tensor;

  auto flattened_from_probs = from_probs.view({-1});
  auto selected_from_prob =
      flattened_from_probs[from.first * from_probs.size(1) + from.second];
  auto selected_direction_prob = direction_probs[static_cast<int>(direction)];

  auto entropy =
      -torch::sum(from_probs * torch::log(from_probs + 1e-10)) -
      torch::sum(direction_probs * torch::log(direction_probs + 1e-10));
  double entropy_coefficient = 0.01;

  auto loss =
      -(torch::log(selected_from_prob) + torch::log(selected_direction_prob)) *
          advantage -
      entropy_coefficient * entropy;

  return loss;
}

void interactive_train() {
  using namespace generals;

  spdlog::info("Starting interactive training");

  auto device = get_device();
  GeneralsNetwork network;
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(1e-3));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> map_size{18, 25};

  spdlog::info("Loading network from {}", NETWORK_PATH);
  try {
    torch::load(network, NETWORK_PATH);
  } catch (const c10::Error &) {
    spdlog::info("Failed to load network, creating a new one");
  }

  while (true) {
    Game game{map_size(gen), map_size(gen), 2};

    spdlog::info("Starting new game with map size {}x{}", game.board.extent(0),
                 game.board.extent(1));

    const auto interact = [&] {
      const auto &player_board = game.player_view(1);
      auto [from_probs, direction_probs] = network->forward(
          player_board.to_tensor(), player_board.action_mask());
      const auto &[from, direction] =
          select_action(from_probs, direction_probs);

      game.apply_inplace({1, from, direction});
      game.next_turn();

      auto loss =
          loss_calculate(1, game, from_probs, direction_probs, from, direction);

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    };

    auto closed = interaction::interaction(game, interact);

    if (closed) {
      spdlog::info("Window closed, saving network to {}", NETWORK_PATH);

      if (std::filesystem::create_directories(DATA_DIR))
        spdlog::info("Created directory {}", DATA_DIR);

      torch::save(network, NETWORK_PATH);
      break;
    }
  }
}

void train(int game_nums, int max_ticks) {
  using namespace generals;

  spdlog::info("Starting training");

  auto device = get_device();
  GeneralsNetwork network;
  torch::optim::Adam optimizer(
      network->parameters(),
      torch::optim::AdamOptions(1e-3).weight_decay(1e-4));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> map_size{18, 25};
  eval::AlgoEval eval;

  spdlog::info("Loading network from {}", NETWORK_PATH);
  try {
    torch::load(network, NETWORK_PATH);
  } catch (const c10::Error &) {
    spdlog::info("Failed to load network, creating a new one");
  }
  network->to(device);

  for (int i = 0; i < game_nums; ++i) {
    const auto w = map_size(gen), h = map_size(gen);
    spdlog::info("Starting game {} with map size {}x{}", i, w, h);

    Game game{w, h, 2};
    auto view_1 = game.player_view(1);

    while (game.tick < max_ticks && !game.is_over()) {
      auto [from_probs, direction_probs] =
          network->forward(view_1.to_tensor(), view_1.action_mask());
      const auto &[from, direction] =
          select_action(from_probs, direction_probs);

      game.apply_inplace({1, from, direction});
      game.next_turn();

      auto loss =
          loss_calculate(1, game, from_probs, direction_probs, from, direction);

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }
  }

  if (std::filesystem::create_directories(DATA_DIR))
    spdlog::info("Created directory {}", DATA_DIR);

  spdlog::info("Saving network to {}", NETWORK_PATH);
  torch::save(network, NETWORK_PATH);
}

} // namespace generals::train
