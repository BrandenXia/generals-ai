#ifndef GENERALS_INTERACTION_HPP
#define GENERALS_INTERACTION_HPP

#include <functional>
#include <raylib.h>
#include <utility>

#include "game.hpp"

namespace generals::interaction {

using SelectAction =
    std::function<std::pair<game::Coord, game::Step::Direction>(
        const PlayerBoard &)>;

void interaction_train(Game &game, SelectAction select_action,
                       std::function<void()> on_turn_end);

} // namespace generals::interaction

#endif
