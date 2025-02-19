#include "cli.hpp"
#include "train.hpp"

using namespace generals;

int main(int argc, char *argv[]) {
  auto args = cli::parse(argc, argv);

  std::visit(
      [](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, cli::args::Train>)
          train::train(arg.game_nums, arg.max_ticks, arg.network_path);
        else if constexpr (std::is_same_v<T, cli::args::Interactive>)
          train::interactive_train(arg.network_path);
      },
      args);
}
