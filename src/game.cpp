#include "game.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
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

Game::Game(std::uint8_t w, std::uint8_t h, std::uint8_t player_count)
    : player_count(player_count), alive_count(player_count), width(w),
      height(h) {
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
    auto player = Player{static_cast<uint8_t>(players.size() + 1)};
    t.owner = player;
    t.army = 1;

    return {player, {x, y}};
  });

  board = {tiles.data(), w, h};
}

inline constexpr std::array<coord::Offset, 4> directions = {
    {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
void Game::apply(Move move) {
  const auto &from = board[move.from];
  if (from.owner != move.player) return;
  if (from.army <= 1) return;

  const auto &offset = directions[static_cast<std::size_t>(move.direction)];
  const auto to_pos = move.from + offset;

  if (!to_pos.valid(width, height)) return;

  const auto &to = board[to_pos];

  if (to.type == Type::Mountain) return;

  from.army = 1;
  if (to.owner == move.player)
    to.army += from.army - 1; // move all but one army
  else {
    // fight
    const auto army = static_cast<int>(from.army) - static_cast<int>(to.army);
    to.army = static_cast<std::uint32_t>(std::abs(army));

    if (army > 0) {
      const MaybePlayer original_owner = to.owner;
      to.owner = move.player; // player wins

      if (to.type == Type::General) {
        to.type = Type::City;
        get_info(original_owner.to_player().value())->alive = false;
        alive_count--;
      }
    }
  }
}

void Game::next_tick() {
  tick++;

  const auto div_by_25 = tick % 25 == 0;
  std::ranges::for_each(tiles, [&](auto &tile) {
    if ((tile.type == Type::City && tile.has_owner()) ||
        tile.type == Type::General)
      tile.army += 1u;
    if (div_by_25 && tile.type == Type::Blank && tile.has_owner())
      tile.army += 1u;
  });
}

} // namespace generals::game
