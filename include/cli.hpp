#ifndef GENERALS_CLI_HPP
#define GENERALS_CLI_HPP

#include <argparse/argparse.hpp>
#include <filesystem>
#include <variant>

#include "game.hpp"

namespace generals::cli {

namespace args {

struct Train {
  int game_nums;
  int max_ticks;
  std::filesystem::path network_path;
  game::Player player;
};

struct Interactive {
  std::filesystem::path network_path;
  game::Player interact_player;
  game::Player opponent;
};

struct Bidirectional {
  int game_nums;
  int max_ticks;
  std::filesystem::path n1_path;
  std::filesystem::path n2_path;
  game::Player n1_player;
  game::Player n2_player;
};

} // namespace args

using CommandArgs = std::variant<args::Train, args::Interactive,
                                 args::Bidirectional, std::monostate>;

CommandArgs parse(int argc, char *argv[]);

} // namespace generals::cli

#endif
