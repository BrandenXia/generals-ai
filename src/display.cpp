#include "display.hpp"

#include <raylib.h>

#include "game.hpp"

namespace generals::display {

inline constexpr unsigned int FPS = 60;

inline constexpr unsigned int TILE_SIZE = 30;
inline constexpr unsigned int BORDER_SIZE = 2;
inline constexpr unsigned int TILE_WITH_BORDER = TILE_SIZE + BORDER_SIZE;

inline constexpr unsigned int FONT_SIZE = 12;
inline constexpr unsigned int TEXT_OFFSET_X = 1;

inline constexpr std::array<Color, 2> player_colors = {BLUE, RED};
inline const std::array<const char *, 4> symbols = {"", "M", "C", "G"};

void init_window(const Game &game) {
  const auto h = game.board.extent(0);
  const auto w = game.board.extent(1);

  SetTraceLogLevel(LOG_ERROR);
  InitWindow(w * TILE_WITH_BORDER - BORDER_SIZE,
             h * TILE_WITH_BORDER - BORDER_SIZE, "Generals AI GUI");
  SetTargetFPS(FPS);
}

void draw_game(const Game &game) {
  const auto h = game.board.extent(0);
  const auto w = game.board.extent(1);

  ClearBackground(LIGHTGRAY);
  for (unsigned int i = 0; i < h; i++)
    for (unsigned int j = 0; j < w; j++) {
      const auto &tile = game.board[i, j];
      const auto offsetI = i * TILE_WITH_BORDER;
      const auto offsetJ = j * TILE_WITH_BORDER;

      DrawRectangle(offsetJ, offsetI, TILE_SIZE, TILE_SIZE,
                    tile.owner.has_value() ? player_colors[tile.owner.value()]
                                           : RAYWHITE);
      switch (tile.type) {
      case game::Type::Mountain:
      case game::Type::City:
      case game::Type::General:
        DrawText(symbols[static_cast<int>(tile.type)], offsetJ + TEXT_OFFSET_X,
                 offsetI, FONT_SIZE, BLACK);
        break;
      case game::Type::Blank:
      case game::Type::Unknown:
      case game::Type::UnknownObstacles:
        break;
      }

      if (tile.army > 0)
        DrawText(std::to_string(tile.army).c_str(), offsetJ + TEXT_OFFSET_X,
                 offsetI + TILE_SIZE - FONT_SIZE, FONT_SIZE, BLACK);
    }
}

std::pair<game::Coord, Rectangle> get_tile_rect(const Game &game,
                                                Vector2 mouse_pos) {
  const auto j = static_cast<int>(mouse_pos.x / TILE_WITH_BORDER),
             i = static_cast<int>(mouse_pos.y / TILE_WITH_BORDER);
  const auto rect =
      Rectangle{static_cast<float>(j * TILE_WITH_BORDER),
                static_cast<float>(i * TILE_WITH_BORDER), TILE_SIZE, TILE_SIZE};
  return {{i, j}, rect};
}

} // namespace generals::display
