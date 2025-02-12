#ifndef GENERALS_GAME_HPP
#define GENERALS_GAME_HPP

#include <ATen/core/TensorBody.h>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <optional>
#include <ostream>
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
  at::Tensor to_tensor() const;
  at::Tensor action_mask() const;
};

using Coord = std::pair<unsigned int, unsigned int>;

struct Step {
  Player player;
  Coord from;
  enum class Direction : std::uint8_t { Up, Down, Left, Right } direction;
};

inline constexpr std::array<std::string, 4> direction_str = {"Up", "Down",
                                                             "Left", "Right"};
inline auto format_as(const Step::Direction &direction) {
  return direction_str[static_cast<int>(direction)];
}

struct Game {
  std::vector<Tile> tiles;
  Board board;
  unsigned int tick;
  unsigned short total_player;
  unsigned short current_player;

  explicit Game(unsigned int width, unsigned int height,
                unsigned int player_count);
  Game(const Game &);
  PlayerBoard player_view(Player player) const;
  Game apply(const Step &step) const;
  void apply_inplace(const Step &step);
  void next_turn();
  inline bool is_over() const { return current_player <= 1; }
};

} // namespace generals::game

namespace generals {

using game::Game, game::PlayerBoard;

}

std::ostream &operator<<(std::ostream &os, generals::game::Tile tile);

template <typename T>
  requires std::is_same_v<T, generals::game::Board> ||
           std::is_same_v<T, generals::PlayerBoard>
std::ostream &operator<<(std::ostream &os, T board) {
  const auto h = board.extent(0);
  const auto w = board.extent(1);
  for (std::size_t i = 0; i < h; ++i)
    for (std::size_t j = 0; j < w; ++j)
      os << board[i, j] << (j == w - 1 ? '\n' : ' ');
  return os;
}

#endif
