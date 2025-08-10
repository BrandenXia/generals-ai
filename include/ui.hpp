#ifndef GENERALS_UI_HPP
#define GENERALS_UI_HPP

#include <raylib.h>
#include <utility>

#include "game.hpp"

namespace generals::ui {

void init_window(const Game &game);

void draw_frame(const Game &game);

std::pair<game::coord::Pos, Rectangle> get_tile_rect(const Game &game,
                                                     Vector2 mouse_pos);

void ui_loop(const Game &game, std::function<void()> callback);

} // namespace generals::ui

#endif
