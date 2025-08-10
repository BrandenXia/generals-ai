#include <raylib.h>

#include "game.hpp"
#include "interaction.hpp"

using namespace generals;

int main() {
  Game game{25, 25};

  interaction::interaction(game, {1}, [] {});
}
