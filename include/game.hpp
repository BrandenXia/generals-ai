#ifndef GENERALS_GAME_H
#define GENERALS_GAME_H

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <functional>
#include <mdspan>
#include <optional>
#include <type_traits>
#include <vector>

namespace generals::game {

namespace coord {

using offset_t = std::int8_t;

struct Offset {
  offset_t x;
  offset_t y;

  inline constexpr Offset operator+(const Offset &other) const {
    return {static_cast<offset_t>(x + other.x),
            static_cast<offset_t>(y + other.y)};
  }
  inline constexpr Offset operator-() const {
    return {static_cast<offset_t>(-x), static_cast<offset_t>(-y)};
  }
  inline constexpr Offset operator-(const Offset &other) const {
    return this->operator+(-other);
  }
  inline constexpr Offset operator*(int factor) const {
    return {static_cast<offset_t>(x * factor),
            static_cast<offset_t>(y * factor)};
  }
  inline constexpr bool operator==(const Offset &other) const {
    return x == other.x && y == other.y;
  }
  inline constexpr bool operator!=(const Offset &other) const {
    return !(*this == other);
  }
};

using pos_t = std::uint8_t;

struct Pos {
  pos_t x;
  pos_t y;

  inline constexpr bool valid(pos_t w, pos_t h) const { return x < w && y < h; }

  inline constexpr Pos operator+(const Offset &offset) const {
    return {static_cast<pos_t>(x + offset.x), static_cast<pos_t>(y + offset.y)};
  }
  inline constexpr Pos operator-(const Offset &offset) const {
    return {static_cast<pos_t>(x - offset.x), static_cast<pos_t>(y - offset.y)};
  }
};

} // namespace coord

enum class Type : std::uint8_t { Blank, Mountain, City, General };

struct Player;
struct MaybePlayer;

struct Player {
  std::uint8_t id;

  inline constexpr operator MaybePlayer() const;
  inline constexpr bool operator==(const Player &p) const { return id == p.id; }
};

struct MaybePlayer {
  std::uint8_t id;

  inline constexpr MaybePlayer(std::uint8_t id = 0) : id(id) {}

  inline constexpr bool has_player() const;
  inline constexpr std::optional<Player> to_player() const;
  inline constexpr operator std::uint32_t() const { return id; }
  inline constexpr bool operator==(const MaybePlayer &p) const {
    return id == p.id;
  }
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

  inline constexpr operator T() const {
    return static_cast<T>((data & mask) >> P);
  }
  inline constexpr void operator=(T value) const {
    auto &ref = data.get();
    ref = (ref & ~mask) | ((static_cast<std::uint32_t>(value) << P) & mask);
  }
  inline constexpr bool operator==(T value) const {
    return operator T() == value;
  }

#define INTEGRAL_OP(op)                                                        \
  template <std::integral I>                                                   \
    requires std::is_integral_v<T>                                             \
  inline constexpr void operator op##=(I i) const {                            \
    operator=(static_cast<T>(operator T() op i));                              \
  }
  INTEGRAL_OP(+)
  INTEGRAL_OP(-)
  INTEGRAL_OP(*)
#undef INTEGRAL_OP
};

} // namespace

// Tile = Type (2 bits) | player (3 bits) | army (everything else)
struct Tile {
private:
  std::uint32_t data;

public:
  coord::Pos pos;

  template <typename T, std::size_t S, std::size_t P>
  using Accessor = TileDataAccessor<T, S, P>;
  Accessor<Type, TILE_TYPE_LEN, 0> type{data};
  Accessor<MaybePlayer, TILE_PLAYER_LEN, TILE_TYPE_LEN> owner{data};
  Accessor<std::uint32_t, TILE_ARMY_LEN, TILE_PLAYER_LEN + TILE_TYPE_LEN> army{
      data};

  inline constexpr Tile(Type type, MaybePlayer player, std::uint32_t army,
                        coord::Pos p)
      : data(static_cast<std::uint32_t>(type) |
             (static_cast<std::uint32_t>(player.id) << 2) | (army << 6)),
        pos(p) {}
  inline constexpr Tile(Type type, coord::Pos p)
      : data(static_cast<std::uint32_t>(type)), pos(p) {}
  inline constexpr Tile(coord::Pos p) : Tile{Type::Blank, p} {}

  inline constexpr bool has_owner() const {
    return owner.operator MaybePlayer().has_player();
  }
};

using _Board = std::mdspan<Tile, std::dextents<std::size_t, 2>>;
struct Board : public _Board {
  inline constexpr Board(Tile *data, std::size_t width, std::size_t height)
      : _Board(data, width, height) {}
  inline constexpr Board() : _Board() {}

  inline constexpr auto operator[](coord::Pos pos) const {
    return _Board::operator[](pos.x, pos.y);
  }
};

struct PlayerInfo {
  Player player;
  bool alive = true;
  coord::Pos general;

  inline constexpr PlayerInfo(Player p, coord::Pos g)
      : player(p), general(g) {};
};

struct Move {
  Player player;
  coord::Pos from;
  enum class Direction : std::uint8_t { Up, Left, Down, Right } direction : 2;
};

struct Game {
  std::uint32_t turn = 0;
  std::uint16_t tick;
  std::uint8_t player_count = 2;
  std::uint8_t alive_count = player_count;
  std::uint8_t width, height;

  Board board;
  std::vector<Tile> tiles;
  std::vector<PlayerInfo> players;

  constexpr Game(std::uint8_t width, std::uint8_t height,
                 std::uint8_t player_count = 2);

  void apply(Move move);
  void next_tick();

  inline constexpr auto get_info(const Player &player) {
    return std::ranges::find_if(
        players, [&](const auto &info) { return info.player == player; });
  }
  inline constexpr bool player_alive(const Player &player) const {
    return const_cast<Game *>(this)->get_info(player)->alive;
  }
  inline constexpr bool is_over() const { return alive_count <= 1; }

  inline constexpr void operator+=(Move move) { apply(move); }
  inline constexpr Game operator+(Move move) {
    auto copy = *this;
    operator+=(move);
    return copy;
  }
};

namespace player {

enum class Type { Blank, Mountain, City, General, Unknown, UnknownObstacles };

} // namespace player

} // namespace generals::game

#endif
