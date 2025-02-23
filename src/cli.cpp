#include "cli.hpp"

#include <argparse/argparse.hpp>
#include <iostream>

#include "dirs.hpp"

#define ADD_NETWORK_NAME(name)                                                 \
  name##_cmd.add_argument("NETWORK_NAME")                                      \
      .help("Network name")                                                    \
      .metavar("NETWORK_NAME");

#define ADD_NETWORK_REQUIRED_1 required()
#define ADD_NETWORK_REQUIRED_0 default_value("network")

#define ADD_NETWORK_ARG(name, required)                                        \
  name##_cmd.add_argument("-n", "--network")                                   \
      .ADD_NETWORK_REQUIRED_##required.help("Network name")                    \
      .metavar("NETWORK_NAME");

#define ADD_GAME_NUM(name)                                                     \
  name##_cmd.add_argument("-g", "--games")                                     \
      .default_value(1000)                                                     \
      .help("Number of games to train on")                                     \
      .scan<'d', int>()                                                        \
      .metavar("NUM_GAMES");

#define ADD_MAX_TICK(name)                                                     \
  name##_cmd.add_argument("-t", "--ticks")                                     \
      .default_value(1000)                                                     \
      .help("Maximum number of ticks per game")                                \
      .scan<'d', int>()                                                        \
      .metavar("MAX_TICKS");

#define DESCRIBE(name, str) name##_cmd.add_description(str)
#define SUBPARSER(name) program.add_subparser(name##_cmd)
#define SUB_CMD_USED(name) (program.is_subcommand_used(name##_cmd))

namespace generals::cli {

inline auto to_network_path(const std::string &path) {
  return (DATA_DIR / path).replace_extension(".pt");
}

CommandArgs parse(int argc, char *argv[]) {
  using argparse::ArgumentParser;

  ArgumentParser program{"generals-ai", "0.1.0"};
  program.add_description("generals.io neural-network-based AI bot");

  ArgumentParser create_cmd("create");
  DESCRIBE(create, "Create a new network");
  ADD_NETWORK_NAME(create);
  create_cmd.add_argument("-p", "--player")
      .help("The player the network will control")
      .scan<'d', int>()
      .metavar("PLAYER")
      .required();
  create_cmd.add_argument("-s", "--size")
      .help("The maximum size of the map, format: WIDTHxHEIGHT")
      .metavar("SIZE")
      .action([](std::string str) {
        std::pair<int, int> size;
        std::sscanf(str.c_str(), "%dx%d", &size.first, &size.second);
        return size;
      })
      .required();
  SUBPARSER(create);

  ArgumentParser info_cmd("info");
  DESCRIBE(info, "Show network info");
  ADD_NETWORK_NAME(info);
  SUBPARSER(info);

  ArgumentParser train_cmd("train");
  DESCRIBE(train, "Train the AI");
  ADD_GAME_NUM(train);
  ADD_MAX_TICK(train);
  ADD_NETWORK_ARG(train, 0);
  SUBPARSER(train);

  ArgumentParser interactive_cmd("interactive");
  DESCRIBE(interactive, "Train the AI interactively");
  ADD_NETWORK_ARG(interactive, 0);
  SUBPARSER(interactive);

  ArgumentParser bidirectional_cmd("bidirectional");
  DESCRIBE(bidirectional, "Train the AI in a bidirectional manner");
  ADD_GAME_NUM(bidirectional);
  ADD_MAX_TICK(bidirectional);
  bidirectional_cmd.add_argument("-n1", "--network1")
      .required()
      .help("Network 1 name")
      .metavar("NETWORK_NAME");
  bidirectional_cmd.add_argument("-n2", "--network2")
      .required()
      .help("Network 2 name")
      .metavar("NETWORK_NAME");
  SUBPARSER(bidirectional);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return std::monostate{};
  }

  using namespace args;
  if SUB_CMD_USED (create) {
    auto network_path = create_cmd.get("NETWORK_NAME");
    auto player = create_cmd.get<int>("--player");
    auto max_size = create_cmd.get<std::pair<int, int>>("--size");
    return Create{to_network_path(network_path), player, max_size};
  } else if SUB_CMD_USED (info) {
    auto network_path = info_cmd.get("NETWORK_NAME");
    return Info{to_network_path(network_path)};
  } else if SUB_CMD_USED (train) {
    auto game_nums = train_cmd.get<int>("--games");
    auto max_ticks = train_cmd.get<int>("--ticks");
    auto network_path = train_cmd.get("--network");
    return Train{game_nums, max_ticks, to_network_path(network_path)};
  } else if SUB_CMD_USED (interactive) {
    auto network_path = interactive_cmd.get("--network");
    return Interactive{to_network_path(network_path)};
  } else if SUB_CMD_USED (bidirectional) {
    auto game_nums = bidirectional_cmd.get<int>("--games");
    auto max_ticks = bidirectional_cmd.get<int>("--ticks");
    auto n1_path = bidirectional_cmd.get("--network1");
    auto n2_path = bidirectional_cmd.get("--network2");
    return Bidirectional{game_nums, max_ticks, to_network_path(n1_path),
                         to_network_path(n2_path)};
  } else
    std::cerr << program;

  return std::monostate{};
}

} // namespace generals::cli
