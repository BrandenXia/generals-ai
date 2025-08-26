#ifndef GENERALS_GAME_HPP
#define GENERALS_GAME_HPP

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iterator>
#include <mdspan>
#include <optional>
#include <type_traits>
#include <utility>
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

inline constexpr bool MaybePlayer::has_player() const { return id != 0; }

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

template <typename T, std::uint8_t S, std::uint8_t P>
  requires(sizeof(T) <= sizeof(std::uint32_t))
struct TileDataAccessor {
  using value_type = T;

private:
  inline static constexpr std::uint32_t mask = make_mask(S, P);
  std::reference_wrapper<std::uint32_t> data;

public:
  constexpr TileDataAccessor(std::uint32_t &data) : data(std::ref(data)) {}

  inline constexpr operator value_type() const {
    return static_cast<value_type>((data & mask) >> P);
  }
  inline constexpr void operator=(value_type value) const {
    auto &ref = data.get();
    ref = (ref & ~mask) | ((static_cast<std::uint32_t>(value) << P) & mask);
  }
  inline constexpr bool operator==(value_type value) const {
    return operator value_type() == value;
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

// Tile = army (27 bits) | player (3 bits) | type (2 bits)
struct Tile {
private:
  std::uint32_t data = 0;

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
             (static_cast<std::uint32_t>(player.id) << TILE_TYPE_LEN) |
             (army << (TILE_PLAYER_LEN + TILE_TYPE_LEN))),
        pos(p) {}
  inline constexpr Tile(Type type, coord::Pos p) : Tile{type, 0, 0, p} {}
  inline constexpr Tile(coord::Pos p) : Tile{Type::Blank, p} {}

  inline constexpr Tile(const Tile &other) noexcept
      : data(other.data), pos(other.pos) {}
  inline constexpr Tile(Tile &&other) noexcept
      : data(other.data), pos(std::move(other.pos)) {
    other.data = 0;
  }
  inline constexpr Tile &operator=(const Tile &other) noexcept {
    if (this != &other) {
      data = other.data;
      pos = other.pos;
    }
    return *this;
  }
  inline constexpr Tile &operator=(Tile &&other) noexcept {
    if (this != &other) {
      data = other.data;
      pos = std::move(other.pos);
      other.data = 0;
    }
    return *this;
  }

  inline constexpr bool has_owner() const {
    return owner.operator MaybePlayer().has_player();
  }
};

using _Board = std::mdspan<Tile, std::dextents<std::size_t, 2>>;
struct Board : public _Board {
  inline constexpr Board(Tile *data, std::size_t width, std::size_t height)
      : _Board(data, width, height) {}
  inline constexpr Board() : _Board() {}

  using _Board::operator[];

  inline constexpr auto operator[](coord::Pos pos) const {
    return operator[](pos.x, pos.y);
  }
  inline constexpr auto &operator[](coord::Pos pos) {
    return operator[](pos.x, pos.y);
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

namespace player {

struct PlayerView;

}

struct Game {
  std::uint32_t turn = 0;
  std::uint16_t tick;
  std::uint8_t player_count = 2;
  std::uint8_t alive_count = player_count;
  std::uint8_t width, height;

  Board board;
  std::vector<Tile> tiles;
  std::vector<PlayerInfo> players;

  Game(std::uint8_t width, std::uint8_t height, std::uint8_t player_count = 2);

  void apply(Move move);
  void next_tick();

  inline constexpr auto get_info(const Player &player) {
    return std::ranges::find_if(
        players, [&](const auto &info) { return info.player == player; });
  }
  inline constexpr bool player_alive(const Player &player) const {
    return const_cast<Game *>(this)->get_info(player)->alive;
  }
  inline constexpr std::optional<Player> is_over() const {
    if (alive_count == 1)
      return std::ranges::find_if(players,
                                  [](const auto &info) { return info.alive; })
          ->player;
    else
      return std::nullopt;
  }
  inline constexpr player::PlayerView player_view(Player player) const;

  inline void operator+=(Move move) { apply(move); }
  inline Game operator+(Move move) const {
    auto copy = *this;
    copy.operator+=(move);
    return copy;
  }
};

namespace player {

enum class Type { Blank, Mountain, City, General, Unknown, UnknownObstacles };

namespace {

inline constexpr std::array<coord::Offset, 8> surround = {
    {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};

#define TILE_ATTR_T(attr) decltype(Tile::attr)::value_type

template <typename T>
struct PlayerTileAttrAccessor {
private:
  const Board &board;
  const Tile &tile;
  Player player;

public:
  inline constexpr PlayerTileAttrAccessor(const Board &b, const Tile &t,
                                          Player p)
      : board(b), tile(t), player(p) {}

#define TILE_ACCESSOR_METH_TEMPLATE(attr)                                      \
  template <typename U = T>                                                    \
    requires std::is_same_v<U, TILE_ATTR_T(attr)>

#define TILE_ACCESSOR_METH_SIG(attr) inline constexpr operator T() const

#define TILE_ACCESSOR_METH(attr)                                               \
  TILE_ACCESSOR_METH_TEMPLATE(attr) TILE_ACCESSOR_METH_SIG(attr) {             \
    return tile.attr;                                                          \
  }

  TILE_ACCESSOR_METH(army)
  TILE_ACCESSOR_METH(owner)

  TILE_ACCESSOR_METH_TEMPLATE(type)
  inline constexpr operator Type() const {
    if (std::ranges::any_of(surround, [&](const auto &offset) {
          const auto pos = tile.pos + offset;
          if (!pos.valid(static_cast<coord::pos_t>(board.extent(0)),
                         static_cast<coord::pos_t>(board.extent(1))))
            return false;

          return board[pos].owner == player;
        }))
      return static_cast<Type>(tile.type.operator game::Type());
    else
      return tile.type == game::Type::Blank ? Type::Unknown
                                            : Type::UnknownObstacles;
  }

#undef TILE_ACCESSOR_METH
#undef TILE_ACCESSOR_METH_SIG
#undef TILE_ACCESSOR_METH_TEMPLATE
};

#define PLAYER_TILE_ATTR_ACCESSOR(attr)                                        \
  PlayerTileAttrAccessor<TILE_ATTR_T(attr)> attr { board, tile, player }

struct PlayerViewTileAccessor {
private:
  const Board &board;
  const Tile &tile;
  Player player;

public:
  inline constexpr PlayerViewTileAccessor(const Board &b, const Tile &t,
                                          Player p)
      : board(b), tile(t), player(p) {}

  inline constexpr auto pos() const { return tile.pos; }

  PLAYER_TILE_ATTR_ACCESSOR(army);
  PLAYER_TILE_ATTR_ACCESSOR(owner);
  PLAYER_TILE_ATTR_ACCESSOR(type);
};

#undef PLAYER_TILE_ATTR_ACCESSOR
#undef TILE_ATTR_T

struct PlayerViewIterator {
  using iterator_category = std::forward_iterator_tag;
  using value_type = PlayerViewTileAccessor;
  using difference_type = std::ptrdiff_t;

private:
  const PlayerView *view;
  std::size_t index;

public:
  inline constexpr PlayerViewIterator() {
    throw std::runtime_error(
        "Default constructor of PlayerView::PlayerViewIterator should not be "
        "used");
  }
  inline constexpr PlayerViewIterator(const PlayerView *view, std::size_t index)
      : view(view), index(index) {}
  inline constexpr PlayerViewIterator(const PlayerViewIterator &) = default;

  inline constexpr PlayerViewIterator &
  operator=(const PlayerViewIterator &) = default;

  inline constexpr value_type operator*() const;
  inline constexpr PlayerViewIterator &operator++() {
    ++index;
    return *this;
  }

  inline constexpr PlayerViewIterator operator++(int) {
    PlayerViewIterator tmp = *this;
    ++*this;
    return tmp;
  }

  inline constexpr bool operator==(const PlayerViewIterator &other) const {
    return index == other.index && view == other.view;
  }
};

static_assert(std::forward_iterator<PlayerViewIterator>);

} // namespace

struct PlayerView {
  const Game *game;
  Player player;

  inline constexpr PlayerView(const Game *g, Player p) : game(g), player(p) {}
  inline constexpr PlayerViewTileAccessor operator[](coord::Pos pos) const {
    return PlayerViewTileAccessor{game->board, game->board[pos], player};
  }
  inline constexpr PlayerViewTileAccessor operator[](coord::pos_t x,
                                                     coord::pos_t y) const {
    return PlayerViewTileAccessor{game->board, game->board[x, y], player};
  }

  inline constexpr PlayerViewIterator begin() const { return {this, 0}; }
  inline constexpr PlayerViewIterator end() const {
    return {this, static_cast<std::size_t>(game->width) * game->height};
  }
};

inline constexpr PlayerViewIterator::value_type
PlayerViewIterator::operator*() const {
  const auto w = view->game->width;
  return (*view)[static_cast<coord::pos_t>(index % w),
                 static_cast<coord::pos_t>(index / w)];
}

} // namespace player

using player::PlayerView;

inline constexpr PlayerView Game::player_view(Player player) const {
  return PlayerView{this, player};
}

} // namespace generals::game

namespace generals {

using game::Game;

}

#endif
