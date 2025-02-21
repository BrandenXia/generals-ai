#include "evaluation.hpp"

namespace generals::eval {

double AlgoEval::operator()(const game::Game &game,
                            const game::Player &player) {
  int enemy_army = 0;
  int player_army = 0;
  int enemy_tiles = 0;
  int player_tiles = 0;

  for (const auto &tile : game.tiles)
    if (tile.owner == player) {
      player_army += tile.army;
      player_tiles++;
    } else if (tile.owner.has_value()) {
      enemy_army += tile.army;
      enemy_tiles++;
    }

  double score =
      (player_army - enemy_army) * 0.001 + (player_tiles - enemy_tiles) * 0.01;

  return score;
}

} // namespace generals::eval
