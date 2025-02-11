#include "interaction.hpp"

#include <optional>

#include "display.hpp"

namespace generals::interaction {

void interaction_train(Game &game, std::function<void()> interact) {
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
      interact();
    }

    EndDrawing();
  }

  CloseWindow();
}

} // namespace generals::interaction
