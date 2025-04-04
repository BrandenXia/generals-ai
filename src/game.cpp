#include "game.hpp"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <utility>

namespace generals::game {

inline constexpr std::array<std::pair<int, int>, 9> nearby = {{{1, 1},
                                                               {1, 0},
                                                               {1, -1},
                                                               {0, 1},
                                                               {0, -1},
                                                               {-1, 1},
                                                               {-1, 0},
                                                               {-1, -1},
                                                               {0, 0}}};

bool PlayerBoard::is_unknown(size_t i, size_t j) const {
  return !std::ranges::any_of(nearby, [this, i, j](const auto &d) {
    const auto [di, dj] = d;
    const auto ni = i + di;
    const auto nj = j + dj;

    // check if the tile is out of bounds
    if (ni >= extent(0) || nj >= extent(1) || ni < 0 || nj < 0) return false;

    return Board::operator[](ni, nj).owner == player;
  });
}

inline constexpr Tile UNKNOWN_TILE{Type::Unknown};
inline constexpr Tile UNKNOWN_OBSTACLES_TILE{Type::UnknownObstacles};

const Tile &PlayerBoard::operator[](size_t i, size_t j) const {
  if (is_unknown(i, j))
    return Board::operator[](i, j).type == Type::Blank ? UNKNOWN_TILE
                                                       : UNKNOWN_OBSTACLES_TILE;
  return Board::operator[](i, j);
}

inline unsigned int manhattanDistance(unsigned int x1, unsigned int y1,
                                      unsigned int x2, unsigned int y2) {
  return std::abs(static_cast<int>(x1) - static_cast<int>(x2)) +
         std::abs(static_cast<int>(y1) - static_cast<int>(y2));
}

Game::Game(unsigned int width, unsigned int height, unsigned int player_count) {
  assert(width > 0 && height > 0 && player_count > 0);
  assert(width * height >= 10 * player_count);

  total_player_count = current_player_count = player_count;
  const auto map_size = width * height;
  tiles.resize(map_size, Tile{Type::Blank});

  std::random_device rd;
  std::mt19937 gen{rd()};
  // clang-format off
  std::uniform_int_distribution<unsigned int>
    mountain_n_dist{map_size / 6, map_size / 4},
    city_n_dist{map_size / 35, map_size / 30},
    army_n_dist{30, 50},
    map_dist{0, map_size - 1};
  // clang-format on

  const auto mountain_n = mountain_n_dist(gen);
  const auto city_n = city_n_dist(gen);

  // generate mountains and cities
  for (unsigned int i = 0; i < mountain_n; ++i)
    tiles[map_dist(gen)] = Tile{Type::Mountain};
  for (unsigned int i = 0; i < city_n; ++i)
    tiles[map_dist(gen)] = Tile{Type::City, std::nullopt, army_n_dist(gen)};

  // generate generals, ensuring they are far enough from each other
  generals_pos.reserve(player_count);
  const unsigned int min_distance = width * height / player_count / 15;
  std::generate_n(std::back_inserter(generals_pos), player_count, [&] {
    unsigned int pos, x, y;
    do {
      pos = map_dist(gen);
      x = pos % width;
      y = pos / width;
    } while (std::ranges::any_of(generals_pos, [&](const auto &p) {
      return manhattanDistance(x, y, p.first, p.second) < min_distance;
    }));

    tiles[pos] = Tile{Type::General, generals_pos.size(), 1};
    return std::make_pair(x, y);
  });

  board = Board{tiles.data(), width, height};
}

Game::Game(const Game &other) : tiles(other.tiles) {
  board = Board{tiles.data(), other.board.extent(0), other.board.extent(1)};
}

PlayerBoard Game::player_view(Player player) const {
  return PlayerBoard{board, player};
}

// Up, Down, Left, Right
inline constexpr std::array<std::pair<int, int>, 4> directions = {
    {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

Game Game::apply(const Step &s) const {
  Game new_game = *this;
  new_game.apply_inplace(s);
  return new_game;
}

void Game::apply_inplace(const Step &step) {
  auto &o = board[step.from.first, step.from.second];
  if (o.owner != step.player) return;
  if (o.army <= 1) return;

  const auto d = directions[static_cast<int>(step.direction)];
  const auto t_pos =
      std::make_pair(step.from.first + d.first, step.from.second + d.second);

  if (t_pos.first >= board.extent(0) || t_pos.second >= board.extent(1) ||
      t_pos.first < 0 || t_pos.second < 0)
    return;

  auto &t = board[t_pos.first, t_pos.second];

  if (t.type == Type::Mountain) return;

  if (t.owner == step.player) {
    // same owner, move army
    t.army += o.army - 1;
  } else {
    // fight
    const int army = static_cast<int>(o.army) - 1 - static_cast<int>(t.army);
    t.army = std::abs(army);
    // taken over
    if (army > 0) {
      t.owner = step.player;

      // if the tile is a general, change it to a city
      if (t.type == Type::General) {
        t.type = Type::City;
        current_player_count--;
      }
    }
  }
  o.army = 1;
}

void Game::next_turn() {
  tick++;

  const auto div_by_2 = tick % 2 == 0;
  const auto div_by_25 = tick % 25 == 0;
  std::ranges::for_each(tiles, [&](auto &tile) {
    if (div_by_2 && (tile.type == Type::City && tile.owner.has_value() ||
                     tile.type == Type::General))
      tile.army++;
    if (div_by_25 && tile.type == Type::Blank && tile.owner.has_value())
      tile.army++;
  });
}

} // namespace generals::game
