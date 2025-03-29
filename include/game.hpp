#ifndef GENERALS_GAME_HPP
#define GENERALS_GAME_HPP

#include <ATen/core/TensorBody.h>
#include <cstddef>
#include <cstdint>
#include <format>
#include <magic_enum/magic_enum.hpp>
#include <mdspan>
#include <optional>
#include <vector>

namespace generals::game {

using Player = std::optional<uint8_t>;

enum class Type {
  Blank,
  Mountain,
  City,
  General,
  // Only for player view
  Unknown,
  UnknownObstacles
};
constexpr inline std::uint8_t type_count = magic_enum::enum_count<Type>();

struct Tile {
  Type type;
  unsigned int army;
  Player owner;

  constexpr explicit Tile(Type t) : type(t), army(0), owner(std::nullopt) {}
  explicit Tile(Type t, Player o, unsigned int a)
      : type(t), army(a), owner(o) {}
};

using Board = std::mdspan<Tile, std::dextents<std::size_t, 2>>;
;

struct PlayerBoard : Board {
  Player player;

  bool is_unknown(std::size_t i, std::size_t j) const;
  const Tile &operator[](std::size_t i, std::size_t j) const;
};

using Coord = std::pair<unsigned int, unsigned int>;

struct Step {
  Player player;
  Coord from;
  enum class Direction : std::uint8_t { Up, Left, Down, Right } direction;
};

inline constexpr std::array<std::string, 4> direction_str = {"Up", "Left",
                                                             "Down", "Right"};
inline auto format_as(const Step::Direction &direction) {
  return direction_str[static_cast<int>(direction)];
}

struct Game {
  std::vector<Tile> tiles;
  Board board;
  unsigned int tick;
  unsigned short total_player_count;
  unsigned short current_player_count;
  std::vector<Coord> generals_pos;

  explicit Game(unsigned int width, unsigned int height,
                unsigned int player_count);
  Game(const Game &);
  PlayerBoard player_view(Player player) const;
  Game apply(const Step &step) const;
  void apply_inplace(const Step &step);
  void next_turn();
  inline bool player_alive(Player player) const {
    return player.has_value() &&
           board[generals_pos[*player].first, generals_pos[*player].second]
                   .type == Type::General;
  }
  inline bool is_over() const { return current_player_count <= 1; }
};

} // namespace generals::game

namespace generals {

using game::Game, game::PlayerBoard;

}

template <>
struct std::formatter<generals::game::Player, char> {
  template <class ParseContext>
  constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <class FormatContext>
  auto format(generals::game::Player player, FormatContext &ctx) const {
    return std::format_to(ctx.out(), "{}",
                          player.has_value() ? static_cast<int>(*player) : -1);
  }
};

#endif
