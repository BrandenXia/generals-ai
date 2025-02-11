#ifndef GENERALS_INTERACTION_HPP
#define GENERALS_INTERACTION_HPP

#include <functional>
#include <raylib.h>
#include <utility>

#include "game.hpp"

namespace generals::interaction {

using Action = std::pair<game::Coord, game::Step::Direction>;

bool interaction_train(Game &game, std::function<void()> interact);

} // namespace generals::interaction

#endif
