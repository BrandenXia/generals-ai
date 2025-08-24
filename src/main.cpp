#include <raylib.h>

#include "game.hpp"
#include "interaction.hpp"
#include "search/heuristics.hpp"

using namespace generals;

int main() {
  Game game{25, 25};
  auto evaluator = eval::hce::Evaluator{};
  search::heuristics::Searcher searcher{std::move(evaluator)};
  const auto player_view = game.player_view({2});

  interaction::interaction(game, {1}, [&] {
    const auto move = searcher(player_view);
    game += move;
  });
}
