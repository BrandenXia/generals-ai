#ifndef GENERALS_EVAL_HPP
#define GENERALS_EVAL_HPP

#include <utility>

#include "game.hpp"

namespace generals::eval {

using score_t = double;

template <typename T>
concept Evaluator = requires(T eval, game::Game game, game::Player player) {
  {
    eval(game, player)
  } -> std::convertible_to<score_t>; // gives possiblity of this player winning,
                                     // [0, 1]
  {
    eval(game, std::declval<game::Move>(), player)
  } -> std::convertible_to<score_t>;
};

template <typename T>
struct EvaluatorBase {
  score_t operator()(game::Game game, game::Player player) {
    return static_cast<T *>(this)->eval(game, player);
  }
  score_t operator()(game::Game game, game::Move move, game::Player player) {
    return operator()(game + move, player) - operator()(game, player);
  }
};

} // namespace generals::eval

#endif
