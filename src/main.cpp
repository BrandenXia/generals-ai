#include <iostream>

#include "game.hpp"

int main() {
  generals::game::Game game{20, 20, 3};
  std::cout << game.player_view(generals::game::Player{1}) << std::endl;
}
