#include "eval.hpp"

#include <spdlog/spdlog.h>

#include "game.hpp"

namespace generals::eval::hce {

score_t Evaluator::operator()(const game::PlayerView &view) {
  const auto player = view.player;

  const auto has_player_owner = [&player](const auto &tile) {
    return tile.owner.operator game::MaybePlayer() == player;
  };

  auto w = view.game->width, h = view.game->height;

  auto player_tile_count = std::ranges::count_if(view, has_player_owner);

  return static_cast<double>(player_tile_count) / (w * h);
}

score_t Evaluator::operator()(const game::PlayerView &view, game::Move move) {
  auto game_copy = *view.game + move;
  auto new_view = game_copy.player_view(view.player);

  auto score = (*this)(view);
  auto new_score = (*this)(new_view);

  spdlog::trace("Evaluating move: {}, score: {} -> {}", move, score, new_score);

  return new_score - score;
}

} // namespace generals::eval::hce
