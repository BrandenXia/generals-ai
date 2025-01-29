#ifndef GENERALS_GAME_HPP
#define GENERALS_GAME_HPP

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
};

struct Step {
  Player player;
  std::pair<unsigned int, unsigned int> from;
  enum class Direction { Up, Down, Left, Right } direction;
};

struct Game {
  std::vector<Tile> tiles;
  Board board;

  explicit Game(unsigned int width, unsigned int height, unsigned int players);
  Game(const Game &);
  PlayerBoard player_view(Player player) const;
  Game apply(const Step &step) const;
  void apply_inplace(const Step &step);
};

} // namespace generals::game

std::ostream &operator<<(std::ostream &os, generals::game::Tile tile);

template <typename T>
  requires std::is_same_v<T, generals::game::Board> ||
           std::is_same_v<T, generals::game::PlayerBoard>
std::ostream &operator<<(std::ostream &os, T board) {
  for (std::size_t i = 0; i < board.extent(0); ++i)
    for (std::size_t j = 0; j < board.extent(1); ++j)
      os << board[i, j] << (j == board.extent(1) - 1 ? '\n' : ' ');
  return os;
}

#endif
