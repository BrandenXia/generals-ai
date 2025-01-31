#include "evaluation.hpp"

namespace generals::eval {

double AlgoEval::operator()(const game::Game &game,
                            const game::Player &player) {
  double score = 0;

  for (const auto &tile : game.tiles)
    if (tile.owner == player) score += tile.army * 0.5 + 1;

  score += (game.total_player - game.current_player) * 100;

  return score;
}

} // namespace generals::eval
