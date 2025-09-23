#include <raylib.h>
#include <spdlog/spdlog.h>

#include "game.hpp"
#include "interaction.hpp"
#include "search/heuristics.hpp"

using namespace generals;

int main() {
  spdlog::set_level(spdlog::level::trace);

  Game game{25, 25};
  auto evaluator = eval::hce::Evaluator{};
  search::heuristics::Searcher searcher{std::move(evaluator)};
  const auto player_view = game.player_view({2});

  interaction::interaction(game, {1}, [&] {
    const auto move = searcher(player_view);
    spdlog::debug("Move: {}", move);
    game += move;
  });
}
