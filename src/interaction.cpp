#include "interaction.hpp"

#include <optional>
#include <raylib.h>

#include "ui.hpp"

namespace generals::interaction {

void interaction(Game &game, game::Player player,
                 std::function<void()> interact) {
  ui::ui_loop(game, [&] {
    const auto &mouse_pos = GetMousePosition();
    if (mouse_pos.x < 0 || mouse_pos.y < 0) return;

    const auto &[coord, mouse_tile] = ui::get_tile_rect(game, mouse_pos);
    DrawRectangleLinesEx(mouse_tile, 2, BLACK);

    std::optional<game::Move::Direction> direction;
    switch (GetKeyPressed()) {
    case KEY_A:
    case KEY_LEFT:
      direction = game::Move::Direction::Left;
      break;
    case KEY_D:
    case KEY_RIGHT:
      direction = game::Move::Direction::Right;
      break;
    case KEY_W:
    case KEY_UP:
      direction = game::Move::Direction::Up;
      break;
    case KEY_S:
    case KEY_DOWN:
      direction = game::Move::Direction::Down;
      break;
    }
    if (direction.has_value()) {
      game::Move move{player, coord, *direction};
      game += move;
      game.next_tick();
      interact();
    }
  });
}

} // namespace generals::interaction
