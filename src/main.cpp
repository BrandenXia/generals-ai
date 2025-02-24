#include <torch/serialize.h>

#include "cli.hpp"
#include "network.hpp"
#include "train.hpp"

#define SAME_V(ARG) constexpr(std::is_same_v<T, cli::args::ARG>)

using namespace generals;

int main(int argc, char *argv[]) {
  auto args = cli::parse(argc, argv);

  std::visit(
      [](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;

        if SAME_V (Create) {
          GeneralsNetwork network{arg.player, arg.max_size};
          torch::save(network, arg.network_path);
          std::cout << "Network created and saved to " << arg.network_path
                    << std::endl;
        } else if SAME_V (Info)
          std::cout << network::info(arg.network_path) << std::endl;
        else if SAME_V (Train)
          train::train(arg.game_nums, arg.max_ticks, arg.network_path);
        else if SAME_V (Interactive)
          train::interactive_train(arg.network_path);
        else if SAME_V (Bidirectional)
          train::bidirectional_train(arg.game_nums, arg.max_ticks, arg.n1_path,
                                     arg.n2_path);
        else
          std::cerr << "No valid command provided" << std::endl;
      },
      args);
}
