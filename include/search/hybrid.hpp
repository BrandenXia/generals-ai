#ifndef GENERALS_SEARCH_HYBRID_HPP
#define GENERALS_SEARCH_HYBRID_HPP

#include "eval.hpp"

namespace generals::search::hybrid {

template <eval::Evaluator E>
struct HybridSearch {
  E evaluator;
};

} // namespace generals::search::hybrid

namespace generals::search {

using hybrid::HybridSearch;

}

#endif
