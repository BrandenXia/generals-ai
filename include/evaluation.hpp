#ifndef GENERALS_EVALUATION_HPP
#define GENERALS_EVALUATION_HPP

#include "game.hpp"

namespace generals::eval {

using Evaluation = std::function<double(const Game &, const game::Player &)>;

struct AlgoEval {
  static double operator()(const Game &game, const game::Player &player);
};

} // namespace generals::eval

namespace generals {

using eval::Evaluation;

}

#endif
