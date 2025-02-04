#include "interaction.hpp"

#include <optional>

#include "display.hpp"

namespace generals::interaction {

void interaction_train(
    Game &game,
    std::function<
        std::pair<game::Coord, game::Step::Direction>(const PlayerBoard &)>
        select_action,
    std::function<void()> on_turn_end) {
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
      on_turn_end();
    }

    EndDrawing();
  }

  CloseWindow();
}

} // namespace generals::interaction
