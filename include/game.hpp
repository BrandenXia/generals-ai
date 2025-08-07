#ifndef GENERALS_GAME_H
#define GENERALS_GAME_H

#include <cstdint>
#include <functional>
#include <mdspan>
#include <optional>
#include <vector>

namespace generals::game {

struct Offset {
  std::int8_t x;
  std::int8_t y;

  inline constexpr Offset operator+(const Offset &other) const {
    return {static_cast<int8_t>(x + other.x), static_cast<int8_t>(y + other.y)};
  }
  inline constexpr Offset operator-() const {
    return {static_cast<int8_t>(-x), static_cast<int8_t>(-y)};
  }
  inline constexpr Offset operator-(const Offset &other) const {
    return this->operator+(-other);
  }
  inline constexpr Offset operator*(int factor) const {
    return {static_cast<int8_t>(x * factor), static_cast<int8_t>(y * factor)};
  }
  inline constexpr bool operator==(const Offset &other) const {
    return x == other.x && y == other.y;
  }
  inline constexpr bool operator!=(const Offset &other) const {
    return !(*this == other);
  }
};

struct Pos {
  std::uint8_t x;
  std::uint8_t y;

  inline constexpr Pos operator+(const Offset &offset) const {
    return {static_cast<uint8_t>(x + offset.x),
            static_cast<uint8_t>(y + offset.y)};
  }
  inline constexpr Pos operator-(const Offset &offset) const {
    return {static_cast<uint8_t>(x - offset.x),
            static_cast<uint8_t>(y - offset.y)};
  }
};

enum class Type : std::uint8_t { Blank, Mountain, City, General };

struct Player;
struct MaybePlayer;

struct Player {
  std::uint8_t id;

  inline constexpr operator MaybePlayer() const;
};

struct MaybePlayer {
  std::uint8_t id;

  inline constexpr operator std::uint32_t() const { return id; }
  inline constexpr bool has_player() const;
  inline constexpr std::optional<Player> to_player() const;
};

inline constexpr Player::operator MaybePlayer() const { return {id}; }

inline constexpr bool MaybePlayer::has_player() const { return id == 0; }

inline constexpr std::optional<Player> MaybePlayer::to_player() const {
  return id == 0 ? std::nullopt : std::optional<Player>{{id}};
}

namespace {

inline constexpr uint32_t make_mask(int s, int p) {
  return ((1u << s) - 1) << p;
}

inline constexpr std::uint8_t TILE_TYPE_LEN = 2;
inline constexpr std::uint8_t TILE_PLAYER_LEN = 3;
inline constexpr std::uint8_t TILE_ARMY_LEN = 27;

template <typename T, std::size_t S, std::size_t P>
  requires(sizeof(T) <= sizeof(std::uint32_t))
struct TileDataAccessor {
private:
  inline static constexpr std::uint32_t mask = make_mask(S, P);
  std::reference_wrapper<std::uint32_t> data;

public:
  constexpr TileDataAccessor(std::uint32_t &data) : data(data) {}

  inline constexpr operator T() const { return (data & mask) >> P; }
  inline constexpr void operator=(T value) const {
    auto &ref = data.get();
    ref = (ref & ~mask) | ((static_cast<std::uint32_t>(value) << P) & mask);
  }
};

} // namespace

// Tile = Type (2 bits) | player (3 bits) | army (everything else)
struct Tile {
private:
  std::uint32_t data;

public:
  Pos pos;

  template <typename T, std::size_t S, std::size_t P>
  using Accessor = TileDataAccessor<T, S, P>;
  Accessor<Type, TILE_TYPE_LEN, 0> type{data};
  Accessor<MaybePlayer, TILE_PLAYER_LEN, TILE_TYPE_LEN> player{data};
  Accessor<std::uint32_t, TILE_ARMY_LEN, TILE_PLAYER_LEN + TILE_TYPE_LEN> army{
      data};

  inline constexpr Tile(Type type, MaybePlayer player, std::uint32_t army,
                        Pos p)
      : data(static_cast<std::uint32_t>(type) |
             (static_cast<std::uint32_t>(player.id) << 2) | (army << 6)),
        pos(p) {}
  inline constexpr Tile(Type type, Pos p)
      : data(static_cast<std::uint32_t>(type)), pos(p) {}
  inline constexpr Tile(Pos p) : Tile{Type::Blank, p} {}
};

using Board = std::mdspan<Tile, std::dextents<std::size_t, 2>>;

struct PlayerInfo {
  Player player;
  bool alive = true;
  Pos general;

  inline constexpr PlayerInfo(Player p, Pos g) : player(p), general(g) {};
};

struct Game {
  std::vector<Tile> tiles;
  Board board;
  std::uint8_t player_count = 2;
  std::uint32_t turn = 0;
  std::uint8_t width;
  std::uint8_t height;
  std::vector<PlayerInfo> players;

  constexpr Game(std::uint8_t width, std::uint8_t height,
                 std::uint8_t player_count = 2);
};

namespace player {

enum class Type { Blank, Mountain, City, General, Unknown, UnknownObstacles };

} // namespace player

} // namespace generals::game

#endif
