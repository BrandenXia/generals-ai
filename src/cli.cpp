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
  program.add_subparser(train_command);

  argparse::ArgumentParser interactive_command("interactive");
  interactive_command.add_description("Interactive training");
  interactive_command.add_argument("-n", "--network")
      .default_value("network")
      .help("Network name")
      .metavar("NETWORK_NAME");
  interactive_command.add_argument("-i", "--interact")
      .default_value(0)
      .help("The player the human will control")
      .scan<'d', int>()
      .metavar("INTERACT_PLAYER");
  interactive_command.add_argument("-o", "--opponent")
      .default_value(1)
      .help("The player the AI will control")
      .scan<'d', int>()
      .metavar("OPPONENT");
  program.add_subparser(interactive_command);

  argparse::ArgumentParser bidirectional_command("bidirectional");
  bidirectional_command.add_description("Bidirectional training");
  bidirectional_command.add_argument("-g", "--games")
      .default_value(1000)
      .help("Number of games to train on")
      .scan<'d', int>()
      .metavar("NUM_GAMES");
  bidirectional_command.add_argument("-t", "--ticks")
      .default_value(1000)
      .help("Maximum number of ticks per game")
      .scan<'d', int>()
      .metavar("MAX_TICKS");
  bidirectional_command.add_argument("-n1", "--network1")
      .required()
      .help("Network 1 name")
      .metavar("NETWORK_NAME");
  bidirectional_command.add_argument("-n2", "--network2")
      .required()
      .help("Network 2 name")
      .metavar("NETWORK_NAME");
  bidirectional_command.add_argument("-p1", "--player1")
      .default_value(0)
      .help("The player the first network will control")
      .scan<'d', int>()
      .metavar("PLAYER");
  bidirectional_command.add_argument("-p2", "--player2")
      .default_value(1)
      .help("The player the second network will control")
      .scan<'d', int>()
      .metavar("PLAYER");
  program.add_subparser(bidirectional_command);

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
    auto interact_player = interactive_command.get<int>("--interact");
    int opponent = interactive_command.get<int>("--opponent");
    return args::Interactive{to_network_path(network_path), interact_player,
                             opponent};
  } else if (program.is_subcommand_used(bidirectional_command)) {
    auto game_nums = bidirectional_command.get<int>("--games");
    auto max_ticks = bidirectional_command.get<int>("--ticks");
    auto n1_path = bidirectional_command.get("--network1");
    auto n2_path = bidirectional_command.get("--network2");
    auto n1_player = bidirectional_command.get<int>("--player1");
    auto n2_player = bidirectional_command.get<int>("--player2");
    return args::Bidirectional{game_nums,
                               max_ticks,
                               to_network_path(n1_path),
                               to_network_path(n2_path),
                               n1_player,
                               n2_player};
  } else
    std::cerr << program;

  return std::monostate{};
}

} // namespace generals::cli
