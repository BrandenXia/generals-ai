#ifndef GENERALS_SEARCH_HPP
#define GENERALS_SEARCH_HPP

#include "eval.hpp"

namespace generals::search {

template <typename T, typename E>
concept Searcher =
    requires(T search, E eval, game::Game game, game::Player player) {
      requires eval::Evaluator<E>;
      { search(game, player, eval) } -> std::convertible_to<game::Move>;
    };

} // namespace generals::search

#endif
