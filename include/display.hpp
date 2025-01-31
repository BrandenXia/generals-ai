#ifndef GENERALS_DISPLAY_HPP
#define GENERALS_DISPLAY_HPP

#include <raylib.h>
#include <utility>

#include "game.hpp"

namespace generals::display {

void init_window(const Game &game);

void draw_game(const Game &game);

std::pair<game::Coord, Rectangle> get_tile_rect(const Game &game,
                                                Vector2 mouse_pos);

} // namespace generals::display

#endif
