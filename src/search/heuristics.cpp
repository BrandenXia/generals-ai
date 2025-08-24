#include "search/heuristics.hpp"

#include <array>
#include <print>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#include "eval.hpp"
#include "game.hpp"

namespace generals::search::heuristics {

using namespace game;
using Direction = Move::Direction;

using TileAccessor =
    decltype(std::declval<PlayerView>()[std::declval<coord::Pos>()]);

constexpr std::array<Direction, 4> directions = {
    {Direction::Up, Direction::Left, Direction::Down, Direction::Right}};
auto get_moves(const TileAccessor &tile) {
  return directions | std::views::transform([&](const auto &dir) {
           return Move{tile.owner.operator MaybePlayer().to_player().value(),
                       tile.pos(), dir};
         });
}

inline constexpr std::array<coord::Offset, 4> directions_offset_map = {
    {{-1, 0}, {0, -1}, {1, 0}, {0, 1}}};
auto legal_moves(const PlayerView &view) {
  using coord::pos_t;

  const auto player = view.player;
  std::println("player.id: {}", player.id);

  const auto player_is_owner = [&player](const auto &tile) {
    std::println("player.id in lambda: {}, owner: {}", player.id,
                 tile.owner.operator MaybePlayer().id);
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
    | std::views::filter(player_is_owner)
    | std::views::transform(get_moves)
    | std::views::join
    | std::views::filter(is_valid_move);
  // clang-format on
}

constexpr int top_k = 5;

Move Searcher::operator()(const PlayerView &view) {
  using MovePair = std::pair<Move, eval::score_t>;
  // clang-format off
  auto moves = legal_moves(view)
    | std::views::transform([&](const auto &move) {
        return std::make_pair(move, (*evaluator)(view, move));
      })
    | std::ranges::to<std::vector<MovePair>>();
  // clang-format on
  std::ranges::sort(moves, std::ranges::greater{}, &MovePair::second);
  auto top_moves = moves | std::views::take(top_k);
  std::println("top_moves.size(): {}", top_moves.size());
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> dist{
      0, std::min<int>(top_k, static_cast<int>(moves.size())) - 1};
  return top_moves[dist(gen)].first;
}

} // namespace generals::search::heuristics
