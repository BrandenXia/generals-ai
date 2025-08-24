#ifndef GENERALS_SEARCH_HEURISTICS_HPP
#define GENERALS_SEARCH_HEURISTICS_HPP

#include <proxy/proxy.h>

#include "eval.hpp"
#include "game.hpp"

namespace generals::search::heuristics {

struct Searcher {
  pro::proxy<eval::Evaluator> evaluator;

  template <typename E>
    requires pro::proxiable_target<E, eval::Evaluator>
  inline constexpr Searcher(E &&e)
      : evaluator(
            pro::make_proxy<eval::Evaluator>(std::forward<decltype(e)>(e))) {}

  game::Move operator()(const game::PlayerView &);
};

} // namespace generals::search::heuristics

#endif
