#ifndef GENERALS_GAME_HPP
#define GENERALS_GAME_HPP

#include <cstddef>
#include <mdspan>
#include <optional>
#include <ostream>
#include <type_traits>
#include <variant>
#include <vector>

namespace generals::game {

struct Player {
  unsigned int id;
  inline bool operator==(const Player &other) const { return id == other.id; }
};
using MaybePlayer = std::optional<Player>;

namespace Tile {

struct Blank {
  MaybePlayer owner;
  unsigned int army;
};

struct Mountain {};

struct City {
  MaybePlayer owner;
  unsigned int army;
};

struct General {
  Player owner;
  unsigned int army;
};

using Type = std::variant<Blank, Mountain, City, General>;

template <typename T>
concept HaveOwner = std::is_same_v<T, Blank> || std::is_same_v<T, City> ||
                    std::is_same_v<T, General>;

template <typename T>
concept HaveOptionalOwner = std::is_same_v<T, Blank> || std::is_same_v<T, City>;

// Unknown tiles are for player's fog of war

// Unknown tiles are just blank tiles with unknown owner
struct Unknown {};

// unknown obstacles cab be either general, city or mountain
struct UnknownObstacles {};

using PlayerView =
    std::variant<Blank, Mountain, City, General, Unknown, UnknownObstacles>;

} // namespace Tile

using Board = std::mdspan<Tile::Type, std::dextents<std::size_t, 2>>;

struct PlayerBoard {
private:
  Board board;
  bool is_unknown(size_t i, size_t j) const;

public:
  Player player;

  PlayerBoard(Board board, Player player);
  inline std::size_t extent(std::size_t i) const { return board.extent(i); }
  Tile::PlayerView operator[](size_t i, size_t j) const;
};

struct Game {
private:
  std::vector<Tile::Type> tiles;

public:
  Board board;

  Game(unsigned int width, unsigned int height, unsigned int players);
  void next();
  PlayerBoard player_view(Player player) const;
};

} // namespace generals::game

#define DEF_OSTREAM(t, c)                                                      \
  inline std::ostream &operator<<(std::ostream &os, generals::game::t tile) {  \
    return os << c;                                                            \
  }

DEF_OSTREAM(Tile::Blank, 'B');
DEF_OSTREAM(Tile::Mountain, 'M');
DEF_OSTREAM(Tile::City, 'C');
DEF_OSTREAM(Tile::General, tile.owner.id);
DEF_OSTREAM(Tile::Unknown, 'U');
DEF_OSTREAM(Tile::UnknownObstacles, 'O');

#undef DEF_OSTREAM

template <typename T>
  requires std::is_same_v<T, generals::game::Tile::Type> ||
           std::is_same_v<T, generals::game::Tile::PlayerView>
std::ostream &operator<<(std::ostream &os, T tile) {
  std::visit([&os](auto &&t) { os << t; }, tile);
  return os;
}

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
