#include "train.hpp"

#include <random>
#include <spdlog/fmt/std.h>
#include <spdlog/spdlog.h>
#include <torch/optim/adam.h>
#include <torch/serialize.h>

#include "dirs.hpp"
#include "evaluation.hpp"
#include "game.hpp"
#include "interaction.hpp"
#include "network.hpp"

namespace generals::train {

inline const auto NETWORK_PATH = DATA_DIR / "network.pt";

void interactive_train() {
  using namespace generals;

  spdlog::info("Starting interactive training");

  GeneralsNetwork network;
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(1e-3));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> map_size{18, 25};
  eval::AlgoEval eval;

  spdlog::info("Try loading network from {}", NETWORK_PATH);
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

      double reward = eval(game, 1);
      auto reward_tensor = torch::tensor(
          {reward}, torch::TensorOptions().dtype(torch::kFloat32));

      auto flattened_from_probs = from_probs.view({-1});
      auto selected_from_prob =
          flattened_from_probs[from.first * from_probs.size(1) + from.second];
      auto selected_direction_prob =
          direction_probs[static_cast<int>(direction)];

      auto loss = -(torch::log(selected_from_prob) +
                    torch::log(selected_direction_prob)) *
                  reward_tensor;

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

} // namespace generals::train
