#include "train.hpp"
#include "device.hpp"

#include <algorithm>
#include <functional>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <limits>
#include <random>
#include <spdlog/fmt/std.h>
#include <spdlog/spdlog.h>
#include <torch/optim/adamw.h>
#include <torch/optim/schedulers/step_lr.h>
#include <torch/serialize.h>

#define FOR_ALIVE_PLAYER(game, idx)                                            \
  for (int idx = 0; idx < game.total_player_count; ++idx)                      \
    if (game.player_alive(idx))

namespace generals::train {

MCTSNode::MCTSNode(const game::Game &s, MCTSNode *p, std::vector<game::Step> a,
                   std::vector<StepMap> probs)
    : state(s), parent(p), actions(a), prior_probs(probs), visits(0) {
  total_values.resize(state.total_player_count, 0.f);
}

MCTSNode::~MCTSNode() {
  for (auto child : children)
    delete child;
}

bool MCTSNode::is_leaf() const { return children.empty(); }

float MCTSNode::ucb_score(game::Player p, float e) const {
  if (visits == 0) return std::numeric_limits<float>::infinity();

  auto exploitation = total_values[p.value()] / visits;

  float prior_probs = 0.f;
  if (parent) {
    const auto &pp_probs = parent->prior_probs[p.value()];
    if (p.value() < actions.size()) {
      auto it = pp_probs.find(actions[p.value()]);
      if (it != pp_probs.end()) prior_probs = it->second;
    }
  }

  float exploration =
      e * prior_probs * std::sqrt(parent ? parent->visits : 1) / (1.f + visits);

  return exploitation + exploration;
}

MCTSNode *MCTSNode::select_child(game::Player p, float e) {
  const auto ucb_cmp = [&p, e](const auto a, const auto b) {
    return a->ucb_score(p, e) < b->ucb_score(p, e);
  };
  return *std::ranges::max_element(children, ucb_cmp);
}

void MCTSNode::expand(GeneralsNetwork &nn) {
  prior_probs.resize(state.total_player_count);

  FOR_ALIVE_PLAYER(state, p) {
    auto general_pos = state.generals_pos[p];
    auto pv = state.player_view(p);
    auto pv_tensor =
        nn->encode(pv, state.tick, general_pos).unsqueeze(0).to(get_device());
    auto mask = network::action_mask(pv_tensor.squeeze(0));

    auto [policy, _] = nn->forward(pv_tensor);
    policy = policy.squeeze(0);

    for (int i = 0; i < policy.size(0); ++i) {
      auto step = nn->idx2step(i);
      if (mask[i].item<float>() == 0) continue;

      prior_probs[p].emplace(step, policy[i].item<float>());
    }
  }

  generate_child();
}

void MCTSNode::generate_child() {
  std::vector<std::vector<game::Step>> all_combin;

  std::function<void(std::vector<game::Step>, int)> generate_combin =
      [&](auto current_combin, auto player_idx) {
        if (player_idx == state.total_player_count) {
          all_combin.push_back(current_combin);
          return;
        };

        if (!state.player_alive(player_idx)) {
          generate_combin(current_combin, player_idx + 1);
          return;
        }

        if (player_idx < prior_probs.size() &&
            prior_probs[player_idx].size() > 0)
          for (const auto &[step, _] : prior_probs[player_idx]) {
            auto next_combin = current_combin;
            next_combin.push_back(step);
            generate_combin(next_combin, player_idx + 1);
          }
        else
          generate_combin(current_combin, player_idx + 1);
      };

  generate_combin({}, 0);

  for (const auto &combin : all_combin) {
    Game next_state = state;
    std::vector<game::Step> child_actions(state.total_player_count);

    for (const auto &step : combin) {
      if (!state.player_alive(step.player)) continue;

      next_state = next_state.apply(step);
      child_actions[step.player.value()] = step;
    }

    next_state.next_turn();
    children.push_back(
        new MCTSNode(next_state, this, child_actions, prior_probs));
  }
}

void MCTSNode::backpropagate(const std::vector<float> &values) {
  visits++;
  for (int p = 0; p < values.size(); ++p)
    if (p < total_values.size()) total_values[p] += values[p];

  if (parent) parent->backpropagate(values);
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
run_mcts(GeneralsNetwork &nn, const game::Game &i_state, int n, float e) {
  auto root = new MCTSNode(i_state);
  root->expand(nn);

  for (int i = 0; i < n; ++i) {
    auto node = root;

    std::vector<game::Player> alive_players;
    for (int p = 0; p < i_state.total_player_count; ++p)
      if (i_state.player_alive(p)) alive_players.push_back(p);

    while (!node->is_leaf()) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<int> dis(0, alive_players.size() - 1);

      node = node->select_child(alive_players[dis(gen)], e);
    }

    std::vector<float> values(i_state.total_player_count, 0.f);

    if (!node->state.is_over()) {
      node->expand(nn);

      for (int p = 0; p < i_state.total_player_count; ++p) {
        if (!node->state.player_alive(p)) continue;

        auto pv = node->state.player_view(p);
        auto general_pos = node->state.generals_pos[p];
        auto pv_tensor = nn->encode(pv, node->state.tick, general_pos)
                             .unsqueeze(0)
                             .to(get_device());
        auto [_, value] = nn->forward(pv_tensor);
        values[p] = value.item<float>();
      }
    } else
      for (int p = 0; p < i_state.total_player_count; ++p)
        if (node->state.player_alive(p))
          values[p] = 1.f;
        else
          values[p] = -1.f;

    node->backpropagate(values);
  }

  std::vector<std::vector<float>> policies(i_state.total_player_count);
  for (int p = 0; p < i_state.total_player_count; ++p)
    if (i_state.player_alive(p))
      policies[p].resize(i_state.board.size() * 4, 0.f);

  for (auto child : root->children) {
    for (int p = 0; p < child->actions.size(); ++p)
      if (i_state.player_alive(p)) {
        const auto &step = child->actions[p];
        int idx = static_cast<int>(step.direction) * i_state.board.size() +
                  step.from.first * i_state.board.extent(0) + step.from.second;
        if (idx < policies[p].size())
          policies[p][idx] = static_cast<float>(child->visits) / root->visits;
      }
  }

  std::vector<float> root_values(i_state.total_player_count, 0.f);
  FOR_ALIVE_PLAYER(i_state, p) {
    root_values[p] = root->total_values[p] / root->visits;
  }

  delete root;
  return {policies, root_values};
}

std::vector<
    std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<float>>>
generate_self_play_data(GeneralsNetwork &nn, int n, int ns, float e) {
  std::vector<
      std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<float>>>
      train_data;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (int i = 0; i < n; i++) {
    game::Game game(nn->max_size.first, nn->max_size.second, 2);
    decltype(train_data) game_data;

    while (!game.is_over()) {
      auto [policies, values] = run_mcts(nn, game, ns, e);

      std::vector<torch::Tensor> policy_tensors(game.total_player_count);
      for (int p = 0; p < policies.size(); ++p)
        if (game.player_alive(p))
          policy_tensors[p] = torch::tensor(policies[p]).to(get_device());

      std::vector<torch::Tensor> pv_tensors(game.total_player_count);
      FOR_ALIVE_PLAYER(game, p) {
        auto pv = game.player_view(p);
        auto general_pos = game.generals_pos[p];
        pv_tensors[p] = nn->encode(pv, game.tick, general_pos)
                            .unsqueeze(0)
                            .to(get_device());
      }

      FOR_ALIVE_PLAYER(game, p) {
        game_data.emplace_back(pv_tensors[p], policy_tensors, values);
      }

      std::vector<game::Step> selected_steps(game.total_player_count);
      FOR_ALIVE_PLAYER(game, p) {
        selected_steps[p] =
            nn->idx2step(policy_tensors[p].argmax().item<int>());
      }

      FOR_ALIVE_PLAYER(game, p) { game = game.apply(selected_steps[p]); }

      game.next_turn();
    }

    std::vector<float> final_values(game.total_player_count, 0.f);
    for (int p = 0; p < game.total_player_count; ++p)
      if (game.player_alive(p))
        final_values[p] = 1.f;
      else
        final_values[p] = -1.f;

    for (auto &[_, __, values] : game_data)
      values = final_values;

    train_data.insert(train_data.end(), game_data.begin(), game_data.end());
    spdlog::info("Game {}/{} finished, data size: {}", i + 1, n,
                 game_data.size());
  }

  return train_data;
};

std::pair<torch::Tensor, torch::Tensor>
calc_loss(const torch::Tensor &predicted_policy,
          const torch::Tensor &predicted_value,
          const torch::Tensor &target_policy, float target_value) {
  auto policy_loss =
      torch::sum(-target_policy * torch::log_softmax(predicted_policy, 1));
  auto target_value_tensor = torch::tensor({target_value}).to(get_device());
  auto value_loss = torch::mse_loss(predicted_value, target_value_tensor);
  return {policy_loss, value_loss};
}

inline void train_step(
    GeneralsNetwork &nn, torch::optim::AdamW &optimizer,
    const std::vector<std::tuple<torch::Tensor, std::vector<torch::Tensor>,
                                 std::vector<float>>> &batch) {
  optimizer.zero_grad();
  auto total_loss = torch::zeros({1}).to(get_device());

  for (const auto &[state, target_policies, target_values] : batch)
    for (int p = 0; p < target_policies.size(); ++p)
      if (state.size(0) > p) {
        auto [predicted_policy, predicted_value] =
            nn->forward(state[p].unsqueeze(0));
        auto [policy_loss, value_loss] =
            calc_loss(predicted_policy, predicted_value, target_policies[p],
                      target_values.at(p));
        total_loss += policy_loss + value_loss;
      }

  total_loss = total_loss / static_cast<int>(batch.size());
  total_loss.backward();
  optimizer.step();
}

void train(std::filesystem::path network_path, unsigned int iter,
           unsigned int game_n, unsigned int mcts_n, float e,
           unsigned int batch_size) {
  GeneralsNetwork nn;
  try {
    nn = network::load(network_path);
  } catch (const std::exception &e) {
    spdlog::error("Network not found");
    return;
  }
  nn->to(get_device());

  std::random_device rd;
  std::mt19937 gen(rd());
  torch::optim::AdamW optimizer(nn->parameters(),
                                torch::optim::AdamWOptions(1e-4));

  for (int i = 0; i < iter; ++i) {
    auto train_data = generate_self_play_data(nn, game_n, mcts_n, e);
    std::shuffle(train_data.begin(), train_data.end(), gen);

    for (int j = 0; j < train_data.size(); j += batch_size) {
      auto end = std::min(j + batch_size,
                          static_cast<unsigned int>(train_data.size()));
      auto batch =
          std::vector(train_data.begin() + j, train_data.begin() + end);
      train_step(nn, optimizer, batch);
    }
  }

  spdlog::info("Training finished, saving network");
  network::save(nn, network_path);
}

} // namespace generals::train
