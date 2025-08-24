#include "eval.hpp"

#include "game.hpp"

namespace generals::eval::hce {

score_t Evaluator::operator()(const game::PlayerView &view) {
  const auto player = view.player;

  const auto has_player_owner = [&player](const auto &tile) {
    return tile.owner.operator game::MaybePlayer() == player;
  };

  auto w = view.game->width, h = view.game->height;

  const double player_tile_count =
      static_cast<double>(std::ranges::count_if(view, has_player_owner));

  return player_tile_count / (w * h);
}

score_t Evaluator::operator()(const game::PlayerView &view, game::Move move) {
  auto game_copy = *view.game + move;
  game::PlayerView new_view = game_copy.player_view(view.player);

  return (*this)(new_view) - (*this)(view);
}

} // namespace generals::eval::hce
