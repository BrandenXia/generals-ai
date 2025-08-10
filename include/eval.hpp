#ifndef GENERALS_EVAL_HPP
#define GENERALS_EVAL_HPP

#include <concepts>

#include "game.hpp"

namespace generals::eval {

using score_t = double;

template <typename T>
concept Evaluator = requires(T eval, game::player::PlayerView view) {
  // gives possiblity of this player winning, [0, 1]
  { eval(view) } -> std::convertible_to<score_t>;
};

} // namespace generals::eval

#endif
