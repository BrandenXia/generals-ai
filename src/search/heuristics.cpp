#include "search/heuristics.hpp"

#include <array>
#include <ranges>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "eval.hpp"
#include "game.hpp"

namespace generals::search::heuristics {

using namespace game;
using Direction = Move::Direction;

using std::declval;
namespace views = std::views;

using TileAccessor = decltype(declval<PlayerView>()[declval<coord::Pos>()]);

constexpr std::array<Direction, 4> directions = {
    {Direction::Up, Direction::Left, Direction::Down, Direction::Right}};
auto get_moves(const TileAccessor &tile) {
  auto player = tile.owner.operator MaybePlayer().to_player().value();
  auto pos = tile.pos();
  return directions | views::transform([=](const auto &dir) {
           return Move{player, pos, dir};
         });
}

inline constexpr std::array<coord::Offset, 4> directions_offset_map = {
    {{-1, 0}, {0, -1}, {1, 0}, {0, 1}}};
auto legal_moves(const PlayerView &view) {
  using coord::pos_t;

  const auto player = view.player;

  const auto player_is_owner = [player](const auto &tile) {
    return tile.owner.operator MaybePlayer() == player;
  };
  const auto is_valid_move = [&view](const Move move) {
    const auto &to_pos =
        move.from +
        directions_offset_map[static_cast<std::size_t>(move.direction)];
    if (!to_pos.valid(view.game->width, view.game->height)) return false;

    const auto &to = view[to_pos];
    return to.type.operator player::Type() != player::Type::Mountain;
  };

  // clang-format off
  return view
    | views::filter(player_is_owner)
    | views::transform(get_moves)
    | views::join
    | views::filter(is_valid_move);
  // clang-format on
}

constexpr int top_k = 5;

Move Searcher::operator()(const PlayerView &view) {
  using MovePair = std::pair<Move, eval::score_t>;
  // clang-format off
  auto moves = legal_moves(view)
    | views::transform([&](const auto &move) {
        return std::make_pair(move, (*evaluator)(view, move));
      })
    | std::ranges::to<std::vector<MovePair>>();
  // clang-format on
  std::ranges::sort(moves, std::ranges::greater{}, &MovePair::second);
  auto top_moves = moves | views::take(top_k);
  return top_moves[0].first;
}

} // namespace generals::search::heuristics
