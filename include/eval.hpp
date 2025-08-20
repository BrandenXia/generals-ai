#ifndef GENERALS_EVAL_HPP
#define GENERALS_EVAL_HPP

#include <proxy/proxy.h>

#include "game.hpp"

namespace generals::eval {

using score_t = double;

// clang-format off
struct Evaluator : pro::facade_builder
  ::add_convention<
      // returns the possibility of winning, [0, 1]
      pro::operator_dispatch<"()">,
      score_t(const game::player::PlayerView &) const
    >
  ::build {};
// clang-format on

} // namespace generals::eval

#endif
