#ifndef GENERALS_SEARCH_HPP
#define GENERALS_SEARCH_HPP

#include "game.hpp"

namespace generals::search {

template <typename T>
concept Searcher = requires(T search, game::player::PlayerView view) {
  { search(view) } -> std::convertible_to<game::Move>;
};

} // namespace generals::search

#endif
