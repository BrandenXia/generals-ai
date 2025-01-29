#include <iostream>

#include "game.hpp"

int main() {
  generals::game::Game game{19, 20, 3};
  std::cout << game.board.extent(0) << " " << game.board.extent(1) << std::endl;
  std::cout << game.board;
  // find general
  std::pair<unsigned int, unsigned int> general_pos;
  for (unsigned int i = 0; i < game.board.extent(0); ++i)
    for (unsigned int j = 0; j < game.board.extent(1); ++j)
      if (game.board[i, j].type == generals::game::Type::General &&
          game.board[i, j].owner == 0) {
        general_pos = {i, j};
        break;
      }
  std::cout << "General position: " << general_pos.first << " "
            << general_pos.second << std::endl;
  std::cout << game.board[general_pos.first, general_pos.second].army
            << std::endl;
  generals::game::Player player{0};
  std::cout << game.player_view(player);

  game.apply_inplace(
      {player, general_pos, generals::game::Step::Direction::Right});
  std::cout << game.board << std::endl;
  std::cout << game.player_view(player);
}
