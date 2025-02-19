#ifndef GENERALS_CLI_HPP
#define GENERALS_CLI_HPP

#include <argparse/argparse.hpp>
#include <filesystem>
#include <variant>

namespace generals::cli {

namespace args {

struct Train {
  int game_nums;
  int max_ticks;
  std::filesystem::path network_path;
};

struct Interactive {
  std::filesystem::path network_path;
};

} // namespace args

using CommandArgs =
    std::variant<args::Train, args::Interactive, std::monostate>;

CommandArgs parse(int argc, char *argv[]);

} // namespace generals::cli

#endif
