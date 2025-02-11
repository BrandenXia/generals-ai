#include "train.hpp"

#include <random>
#include <torch/optim/adam.h>
#include <torch/serialize.h>

#include "evaluation.hpp"
#include "game.hpp"
#include "interaction.hpp"
#include "network.hpp"

namespace generals::train {

void interactive_train() {
  using namespace generals;

  GeneralsNetwork network;
  torch::optim::Adam optimizer(network->parameters(),
                               torch::optim::AdamOptions(1e-3));
  std::random_device rd;
  std::uniform_int_distribution<unsigned int> map_size{18, 25};
  eval::AlgoEval eval;

  try {
    torch::load(network, "network.pt");
  } catch (const c10::Error &) {}

  while (true) {
    Game game{map_size(rd), map_size(rd), 2};

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

    auto closed = interaction::interaction_train(game, interact);

    if (closed) {
      torch::save(network, "network.pt");
      break;
    }
  }
}

} // namespace generals::train
