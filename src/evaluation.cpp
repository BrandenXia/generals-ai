#include "evaluation.hpp"

namespace generals::eval {

double AlgoEval::operator()(const game::Game &game,
                            const game::Player &player) {
  double score = 0;
  int enemy_army = 0;
  int player_army = 0;

  for (const auto &tile : game.tiles)
    if (tile.owner == player) {
      player_army += tile.army;
      score += 0.01;
    } else if (!tile.owner.has_value())
      enemy_army += tile.army;

  score += (player_army - enemy_army) * 0.01;

  return score;
}

} // namespace generals::eval
