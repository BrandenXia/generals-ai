#include "ui.hpp"

#include <raylib.h>

#include "game.hpp"

namespace generals::ui {

inline constexpr unsigned int FPS = 60;

inline constexpr auto TILE_SIZE = 30;
inline constexpr auto BORDER_SIZE = 2;
inline constexpr auto TILE_WITH_BORDER = TILE_SIZE + BORDER_SIZE;

inline constexpr auto FONT_SIZE = 12;
inline constexpr auto TEXT_OFFSET_X = 1;

inline constexpr std::array<Color, 3> player_colors = {BLUE, RED, GREEN};
inline const std::array<const char *, 4> symbols = {"", "M", "C", "G"};

void init_window(const Game &game) {
  SetTraceLogLevel(LOG_ERROR);
  InitWindow(game.width * TILE_WITH_BORDER - BORDER_SIZE,
             game.height * TILE_WITH_BORDER - BORDER_SIZE, "Generals AI GUI");
  SetTargetFPS(FPS);
}

void draw_frame(const Game &game) {
  ClearBackground(LIGHTGRAY);
  for (game::coord::pos_t i = 0; i < game.height; i++)
    for (game::coord::pos_t j = 0; j < game.width; j++) {
      const auto &tile = game.board[i, j];
      const auto offsetI = i * TILE_WITH_BORDER;
      const auto offsetJ = j * TILE_WITH_BORDER;

      DrawRectangle(
          offsetJ, offsetI, TILE_SIZE, TILE_SIZE,
          tile.has_owner()
              ? player_colors[tile.owner.operator game::MaybePlayer().id - 1]
              : RAYWHITE);
      switch (tile.type) {
      case game::Type::Mountain:
      case game::Type::City:
      case game::Type::General:
        DrawText(
            symbols[static_cast<std::size_t>(tile.type.operator game::Type())],
            offsetJ + TEXT_OFFSET_X, offsetI, FONT_SIZE, BLACK);
        break;
      default:
        break;
      }

      if (tile.army > 0)
        DrawText(std::to_string(tile.army).c_str(), offsetJ + TEXT_OFFSET_X,
                 offsetI + TILE_SIZE - FONT_SIZE, FONT_SIZE, BLACK);
    }
}

std::pair<game::coord::Pos, Rectangle> get_tile_rect(const Game &,
                                                     Vector2 mouse_pos) {
  const auto j = static_cast<game::coord::pos_t>(mouse_pos.x /
                                                 TILE_WITH_BORDER),
             i = static_cast<game::coord::pos_t>(mouse_pos.y /
                                                 TILE_WITH_BORDER);
  const Rectangle rect = {static_cast<float>(j * TILE_WITH_BORDER),
                          static_cast<float>(i * TILE_WITH_BORDER), TILE_SIZE,
                          TILE_SIZE};
  return {{i, j}, rect};
}

} // namespace generals::ui
