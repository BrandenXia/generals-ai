#include "game.hpp"

#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

namespace generals::game {

inline unsigned int manhattanDistance(unsigned int x1, unsigned int y1,
                                      unsigned int x2, unsigned int y2) {
  return static_cast<unsigned int>(
      std::abs(static_cast<int>(x1) - static_cast<int>(x2)) +
      std::abs(static_cast<int>(y1) - static_cast<int>(y2)));
}

constexpr Game::Game(std::uint8_t w, std::uint8_t h, std::uint8_t player_count)
    : player_count(player_count), width(w), height(h) {
  const unsigned int map_size = w * h;
  tiles.reserve(map_size);
  for (unsigned int i = 0; i < map_size; ++i)
    tiles.emplace_back(Tile{
        Type::Blank,
        {static_cast<std::uint8_t>(i % w), static_cast<std::uint8_t>(i / w)}});

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
    tiles[map_dist(gen)].type = Type::Mountain;
  for (unsigned int i = 0; i < city_n; ++i) {
    const auto &t = tiles[map_dist(gen)];
    t.type = Type::City;
    t.army = army_n_dist(gen);
  }

  players.reserve(player_count);
  const unsigned int min_distance = width * height / player_count / 15;
  std::generate_n(std::back_inserter(players), player_count, [&] -> PlayerInfo {
    unsigned int pos;
    std::uint8_t x, y;
    do {
      pos = map_dist(gen);
      x = pos % w;
      y = static_cast<std::uint8_t>(pos / w);
    } while (std::ranges::any_of(players, [&](const auto &p) {
      return manhattanDistance(x, y, p.general.x, p.general.y) <= min_distance;
    }));

    const auto &t = tiles[pos];
    t.type = Type::General;
    auto player = Player{static_cast<uint8_t>(players.size())};
    t.player = player;
    t.army = 1;

    return {player, {x, y}};
  });

  board = Board{tiles.data(), w, h};
}

} // namespace generals::game
