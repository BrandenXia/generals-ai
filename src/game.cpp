#include "game.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <type_traits>
#include <utility>

namespace generals::game {

PlayerBoard::PlayerBoard(Board board, Player player)
    : board(board), player(player) {}

inline const std::array<std::pair<int, int>, 9> directions = {{{1, 1},
                                                               {1, 0},
                                                               {1, -1},
                                                               {0, 1},
                                                               {0, -1},
                                                               {-1, 1},
                                                               {-1, 0},
                                                               {-1, -1},
                                                               {0, 0}}};

template <Tile::HaveOwner T>
inline bool check_owner(T tile, Player player) {
  if constexpr (Tile::HaveOptionalOwner<T>)
    return tile.owner.has_value() && tile.owner.value() == player;
  else
    return tile.owner == player;
}

bool PlayerBoard::is_unknown(size_t i, size_t j) const {
  return !std::ranges::any_of(directions, [this, i, j](const auto &d) {
    const auto [di, dj] = d;
    const auto ni = i + di;
    const auto nj = j + dj;

    // check if the tile is out of bounds
    if (ni >= board.extent(0) || nj >= board.extent(1) || ni < 0 || nj < 0)
      return false;

    return std::visit(
        [this](auto &&tile) {
          using T = std::decay_t<decltype(tile)>;

          if constexpr (Tile::HaveOwner<T>)
            return check_owner(tile, player);
          else
            return false;
        },
        board[ni, nj]);
  });
}

Tile::PlayerView PlayerBoard::operator[](size_t i, size_t j) const {
  const auto &tile = board[i, j];

  if (is_unknown(i, j))
    return std::visit(
        [](auto &&tile) -> Tile::PlayerView {
          using T = std::decay_t<decltype(tile)>;

          if constexpr (std::is_same_v<T, Tile::Blank>)
            return Tile::Unknown{};
          else
            return Tile::UnknownObstacles{};
        },
        tile);

  // forward the tile to the player view
  return std::visit(
      [player = player](auto &&tile) -> Tile::PlayerView {
        using T = std::decay_t<decltype(tile)>;

        if constexpr (std::is_same_v<T, Tile::Blank> ||
                      std::is_same_v<T, Tile::Mountain> ||
                      std::is_same_v<T, Tile::City> ||
                      std::is_same_v<T, Tile::General>)
          return tile;

        std::unreachable();
      },
      tile);
}

unsigned int manhattanDistance(unsigned int x1, unsigned int y1,
                               unsigned int x2, unsigned int y2) {
  return std::abs(static_cast<int>(x1) - static_cast<int>(x2)) +
         std::abs(static_cast<int>(y1) - static_cast<int>(y2));
}

Game::Game(unsigned int width, unsigned int height, unsigned int players) {
  assert(width > 0 && height > 0 && players > 0);
  assert(width * height >= 10 * players);

  const auto map_size = width * height;
  tiles.resize(map_size, Tile::Blank{});

  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<unsigned int> mountain_n_dist{map_size / 10,
                                                              map_size / 5},
      city_n_dist{map_size / 35, map_size / 20}, army_n_dist{30, 50},
      map_dist{0, map_size - 1};

  const auto mountain_n = mountain_n_dist(gen);
  const auto city_n = city_n_dist(gen);

  // generate mountains and cities
  for (unsigned int i = 0; i < mountain_n; ++i)
    tiles[map_dist(gen)] = Tile::Mountain{};
  for (unsigned int i = 0; i < city_n; ++i)
    tiles[map_dist(gen)] = Tile::City{std::nullopt, army_n_dist(gen)};

  // generate generals, ensuring they are far enough from each other
  std::vector<std::pair<unsigned int, unsigned int>> general_positions;
  const unsigned int min_distance = width * height / players / 10;
  std::generate_n(std::back_inserter(general_positions), players, [&] {
    unsigned int pos, x, y;
    do {
      pos = map_dist(gen);
      x = pos % width;
      y = pos / width;
    } while (std::ranges::any_of(general_positions, [&](const auto &p) {
      return manhattanDistance(x, y, p.first, p.second) < min_distance;
    }));

    tiles[pos] = Tile::General{Player(general_positions.size()), 1};
    return std::make_pair(x, y);
  });

  board = Board{tiles.data(), width, height};
}

PlayerBoard Game::player_view(Player player) const {
  return PlayerBoard{board, player};
}

} // namespace generals::game
