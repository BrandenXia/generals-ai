#include "cli.hpp"
#include "train.hpp"

using namespace generals;

int main(int argc, char *argv[]) {
  auto args = cli::parse(argc, argv);

  std::visit(
      [](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, cli::args::Train>)
          train::train(arg.game_nums, arg.max_ticks, arg.network_path,
                       arg.player);
        else if constexpr (std::is_same_v<T, cli::args::Interactive>)
          train::interactive_train(arg.network_path, arg.interact_player,
                                   arg.opponent);
        else if constexpr (std::is_same_v<T, cli::args::Bidirectional>)
          train::bidirectional_train(arg.game_nums, arg.max_ticks, arg.n1_path,
                                     arg.n2_path, arg.n1_player, arg.n2_player);
        else
          std::cerr << "No valid command provided" << std::endl;
      },
      args);
}
