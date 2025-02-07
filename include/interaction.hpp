#ifndef GENERALS_INTERACTION_HPP
#define GENERALS_INTERACTION_HPP

#include <functional>
#include <raylib.h>
#include <utility>

#include "game.hpp"

namespace generals::interaction {

using Action = std::pair<game::Coord, game::Step::Direction>;
using SelectAction =
    std::function<std::pair<game::Coord, game::Step::Direction>(
        const PlayerBoard &)>;
using Optimization = std::function<void(Action)>;

void interaction_train(Game &game, SelectAction select_action,
                       Optimization optimization);

} // namespace generals::interaction

#endif
