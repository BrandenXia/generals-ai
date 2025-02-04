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

    const auto &select_action = [&network](const game::PlayerBoard &board) {
      return network.select_action(board);
    };
    const auto on_turn_end = [&]() {
      double reward = eval(game, 1);
      auto loss = -torch::log(torch::scalar_tensor(reward));
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    };

    interaction::interaction_train(game, select_action, on_turn_end);
  }

  return 0;
}
