#ifndef GENERALS_EVAL_HPP
#define GENERALS_EVAL_HPP

#include <proxy/proxy.h>

#include "game.hpp"
#include "proxy/v4/proxy.h"

namespace generals::eval {

using score_t = double;

// clang-format off
struct Evaluator : pro::facade_builder
  ::add_convention<
      // returns the possibility of winning, [0, 1]
      pro::operator_dispatch<"()">,
      score_t(const game::PlayerView &)
    >
  ::add_convention<
      pro::operator_dispatch<"()">,
      score_t(const game::PlayerView &, game::Move)
    >
  ::build {};
// clang-format on

namespace hce {

struct Evaluator {
  score_t operator()(const game::PlayerView &);
  score_t operator()(const game::PlayerView &, game::Move);
};

} // namespace hce

} // namespace generals::eval

#endif
