#ifndef GENERALS_TRAIN_HPP
#define GENERALS_TRAIN_HPP

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "game.hpp"
#include "network.hpp"

namespace generals::train {

struct StepHasher {
  std::size_t operator()(const game::Step &s) const {
    return ((std::hash<unsigned int>()(s.from.first) ^
             (std::hash<unsigned int>()(s.from.second) << 1) >> 1) ^
            std::hash<game::Step::Direction>()(s.direction) << 1) ^
           std::hash<game::Player>()(s.player);
  }
};

using StepMap = std::unordered_map<game::Step, float, StepHasher>;

struct MCTSNode {
  Game state;
  std::vector<game::Step> actions;

  MCTSNode *parent;
  std::vector<MCTSNode *> children;

  int visits;
  std::vector<float> total_values;
  std::vector<StepMap> prior_probs;

  MCTSNode(const Game &s, MCTSNode *p = nullptr, std::vector<game::Step> a = {},
           std::vector<StepMap> probs = {});
  ~MCTSNode();

  bool is_leaf() const;
  float ucb_score(game::Player player, float exploration_constant) const;
  MCTSNode *select_child(game::Player player, float exploration_constant);
  void expand(GeneralsNetwork &network);
  void generate_child();
  void backpropagate(const std::vector<float> &values);
};

std::pair<std::vector<std::vector<float>>, std::vector<float>>
run_mcts(GeneralsNetwork &net, const game::Game &initial_state,
         int num_simulations, float exploration_constant);

std::vector<
    std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<float>>>
generate_self_play_data(GeneralsNetwork &net, int num_games,
                        int num_simulations, float exploration_constant);

std::pair<torch::Tensor, torch::Tensor>
calc_loss(const torch::Tensor &predicted_policy,
          const torch::Tensor &predicted_value,
          const torch::Tensor &target_policy, float target_value);

void train(std::filesystem::path network_path, unsigned int iterations,
           unsigned int game_num, unsigned int mcts_num,
           float exploration_constant, unsigned int batch_size);

} // namespace generals::train

#endif
