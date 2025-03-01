#ifndef GENERALS_CLI_HPP
#define GENERALS_CLI_HPP

#include <argparse/argparse.hpp>
#include <filesystem>
#include <utility>
#include <variant>

#include "game.hpp"

namespace generals::cli {

namespace args {

struct Create {
  std::filesystem::path network_path;
  game::Player player;
  std::pair<int, int> max_size;
};

struct Info {
  std::filesystem::path network_path;
};

struct Train {
  std::filesystem::path network_path;
  unsigned int iterations;
  unsigned int game_num;
  unsigned int mcts_num;
  float exploration_constant;
  unsigned int batch_size;
};

} // namespace args

using CommandArgs =
    std::variant<args::Create, args::Info, args::Train, std::monostate>;

CommandArgs parse(int argc, char *argv[]);

} // namespace generals::cli

#endif
