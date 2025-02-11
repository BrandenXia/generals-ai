#include <ATen/ops/mse_loss.h>
#include <c10/core/TensorOptions.h>
#include <torch/optim/adam.h>

#include "evaluation.hpp"
#include "game.hpp"
#include "interaction.hpp"
#include "network.hpp"

using namespace generals;

int main() {
  GeneralsNetwork network{20, 20};
  torch::optim::Adam optimizer(network.parameters(),
                               torch::optim::AdamOptions(1e-3));
  eval::AlgoEval eval;

  for (int i = 0; i < 100; i++) {
    Game game{20, 20, 2};

    const auto interact = [&] {
      const auto &player_board = game.player_view(1);
      auto [from_probs, direction_probs] =
          network.forward(player_board.to_tensor(), player_board.action_mask());
      const auto &[from, direction] =
          select_action(from_probs, direction_probs);

      std::cout << "from: " << from.first << " " << from.second << std::endl;
      std::cout << "direction: " << static_cast<int>(direction) << std::endl;

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

    interaction::interaction_train(game, interact);
  }

  return 0;
}
