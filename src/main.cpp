#include "game.hpp"
#include "interaction.hpp"

std::pair<generals::game::Coord, generals::game::Step::Direction>
select_action(const generals::game::Game &game,
              const generals::game::Player &player) {
  return {{0, 0}, generals::game::Step::Direction::Up};
}

int main() {
  generals::Game game{20, 20, 2};
  generals::interaction::interaction_loop(game, select_action);

  return 0;
}
