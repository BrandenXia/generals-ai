#include "cli.hpp"

#include <iostream>

#include "dirs.hpp"

namespace generals::cli {

inline auto to_network_path(const std::string &path) {
  return (DATA_DIR / path).replace_extension(".pt");
}

CommandArgs parse(int argc, char *argv[]) {
  argparse::ArgumentParser program{"generals-ai", "0.1.0"};
  program.add_description("generals.io neural-network-based AI bot");

  argparse::ArgumentParser train_command("train");
  train_command.add_description("Train the AI");
  train_command.add_argument("-g", "--games")
      .default_value(1000)
      .help("Number of games to train on")
      .scan<'d', int>()
      .metavar("NUM_GAMES");
  train_command.add_argument("-t", "--ticks")
      .default_value(1000)
      .help("Maximum number of ticks per game")
      .scan<'d', int>()
      .metavar("MAX_TICKS");
  train_command.add_argument("-n", "--network")
      .default_value("network")
      .help("Network name")
      .metavar("NETWORK_NAME");
  train_command.add_argument("-p", "--player")
      .default_value(1)
      .help("The player the network will control")
      .scan<'d', int>()
      .metavar("PLAYER");

  argparse::ArgumentParser interactive_command("interactive");
  interactive_command.add_description("Interactive training");
  interactive_command.add_argument("-n", "--network")
      .default_value("network")
      .help("Network name")
      .metavar("NETWORK_NAME");

  program.add_subparser(train_command);
  program.add_subparser(interactive_command);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return std::monostate{};
  }

  if (program.is_subcommand_used(train_command)) {
    auto game_nums = train_command.get<int>("--games");
    auto max_ticks = train_command.get<int>("--ticks");
    auto network_path = train_command.get("--network");
    auto player = train_command.get<int>("--player");
    return args::Train{game_nums, max_ticks, to_network_path(network_path),
                       player};
  } else if (program.is_subcommand_used(interactive_command)) {
    auto network_path = interactive_command.get("--network");
    return args::Interactive{to_network_path(network_path)};
  } else
    std::cerr << program;

  return std::monostate{};
}

} // namespace generals::cli
