#ifndef GENERALS_SEARCH_HYBRID_HPP
#define GENERALS_SEARCH_HYBRID_HPP

#include <utility>

#include <proxy/proxy.h>

#include "eval.hpp"

namespace generals::search::hybrid {

struct HybridSearch {
  pro::proxy<eval::Evaluator> evaluator;

  template <typename E>
    requires pro::proxiable<E, eval::Evaluator>
  inline constexpr HybridSearch(E &&e)
      : evaluator(pro::make_proxy(std::forward<E>(e))) {}
};

} // namespace generals::search::hybrid

namespace generals::search {

using hybrid::HybridSearch;

}

#endif
