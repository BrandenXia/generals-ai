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
  int game_nums;
  int max_ticks;
  std::filesystem::path network_path;
};

struct Interactive {
  std::filesystem::path network_path;
};

struct Bidirectional {
  int game_nums;
  int max_ticks;
  std::filesystem::path n1_path;
  std::filesystem::path n2_path;
};

} // namespace args

using CommandArgs =
    std::variant<args::Create, args::Info, args::Train, args::Interactive,
                 args::Bidirectional, std::monostate>;

CommandArgs parse(int argc, char *argv[]);

} // namespace generals::cli

#endif
