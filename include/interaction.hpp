#ifndef GENERALS_INTERACTION_HPP
#define GENERALS_INTERACTION_HPP

#include <functional>

#include "game.hpp"

namespace generals::interaction {

void interaction(Game &game, game::Player player,
                 std::function<void()> interact);

} // namespace generals::interaction

#endif
