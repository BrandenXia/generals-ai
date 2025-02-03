#ifndef GENERALS_INTERACTION_HPP
#define GENERALS_INTERACTION_HPP

#include <optional>
#include <raylib.h>
#include <utility>

#include "display.hpp"
#include "game.hpp"

namespace generals::interaction {

template <typename T>
concept SelectAction = requires(T t) {
  {
    t(std::declval<const PlayerBoard &>())
  } -> std::convertible_to<std::pair<game::Coord, game::Step::Direction>>;
};

template <SelectAction T>
void interaction_loop(Game &game, T select_action) {
  display::init_window(game);

  while (!WindowShouldClose()) {
    BeginDrawing();

    display::draw_game(game);
    const auto &[coord, mouse_tile] =
        display::get_tile_rect(game, GetMousePosition());
    DrawRectangleLinesEx(mouse_tile, 2, BLACK);

    std::optional<game::Step::Direction> direction;
    switch (GetKeyPressed()) {
    case KEY_A:
    case KEY_LEFT:
      direction = game::Step::Direction::Left;
      break;
    case KEY_D:
    case KEY_RIGHT:
      direction = game::Step::Direction::Right;
      break;
    case KEY_W:
    case KEY_UP:
      direction = game::Step::Direction::Up;
      break;
    case KEY_S:
    case KEY_DOWN:
      direction = game::Step::Direction::Down;
      break;
    }
    if (direction.has_value()) {
      game.apply_inplace({0, coord, direction.value()});
      const auto &[coord, direction] = select_action(game.player_view(1));
      game.apply_inplace({1, coord, direction});
      game.next_turn();
    }

    EndDrawing();
  }

  CloseWindow();
}

} // namespace generals::interaction

#endif
