#ifndef GENERALS_EVALUATION_HPP
#define GENERALS_EVALUATION_HPP

#include <concepts>
#include <utility>

#include "game.hpp"

namespace generals::eval {

template <typename T>
concept Evaluation = requires(T t) {
  // player winning chance, should be in [0, 1]
  {
    t(std::declval<const Game &>(), std::declval<const game::Player &>())
  } -> std::convertible_to<double>;
};

struct AlgoEval {
  static double operator()(const Game &game, const game::Player &player);
};
static_assert(Evaluation<AlgoEval>);

} // namespace generals::eval

namespace generals {

using eval::Evaluation;

}

#endif
