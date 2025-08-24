#ifndef GENERALS_SEARCH_HPP
#define GENERALS_SEARCH_HPP

#include <proxy/proxy.h>

#include "game.hpp"

namespace generals::search {

// clang-format off
struct Searcher : pro::facade_builder
  ::add_convention<
      pro::operator_dispatch<"()">,
      game::Move(const game::PlayerView &)
    >
  ::build {};
// clang-format on

} // namespace generals::search

#endif
