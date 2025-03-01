#include "cli.hpp"

#include <argparse/argparse.hpp>
#include <iostream>

#include "dirs.hpp"

#define ADD_NETWORK_NAME(name)                                                 \
  name##_cmd.add_argument("NETWORK_NAME")                                      \
      .help("Network name")                                                    \
      .metavar("NETWORK_NAME");

#define ADD_NETWORK_ARG(name)                                                  \
  name##_cmd.add_argument("-n", "--network")                                   \
      .help("Network name")                                                    \
      .metavar("NETWORK_NAME")                                                 \
      .required();

#define ADD_GAME_NUM(name)                                                     \
  name##_cmd.add_argument("-g", "--games")                                     \
      .default_value(50)                                                       \
      .help("Number of games to train on")                                     \
      .scan<'d', int>()                                                        \
      .metavar("NUM_GAMES");

#define DESCRIBE(name, str) name##_cmd.add_description(str)
#define SUBPARSER(name) program.add_subparser(name##_cmd)
#define SUB_CMD_USED(name) (program.is_subcommand_used(name##_cmd))

namespace generals::cli {

inline auto to_network_path(const std::string &path) {
  return (DATA_DIR / path / path).replace_extension(".pt");
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
  ADD_NETWORK_ARG(train);
  train_cmd.add_argument("-i", "--iterations")
      .default_value(100)
      .help("Number of training iterations")
      .scan<'d', int>()
      .metavar("NUM_ITERATIONS");
  train_cmd.add_argument("-m", "--mcts")
      .default_value(200)
      .help("Number of MCTS simulations per move")
      .scan<'d', int>()
      .metavar("MCTS_NUM");
  train_cmd.add_argument("-e", "--exploration")
      .default_value(1.f)
      .help("Exploration constant for MCTS")
      .scan<'g', float>()
      .metavar("EXPLORATION_CONSTANT");
  train_cmd.add_argument("-b", "--batch")
      .default_value(32)
      .help("Batch size for training")
      .scan<'d', int>()
      .metavar("BATCH_SIZE");
  SUBPARSER(train);

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
    auto network_path = train_cmd.get("--network");
    auto iterations = train_cmd.get<int>("--iterations");
    auto game_num = train_cmd.get<int>("--games");
    auto mcts_num = train_cmd.get<int>("--mcts");
    auto exploration_constant = train_cmd.get<float>("--exploration");
    auto batch_size = train_cmd.get<int>("--batch");
    return Train{to_network_path(network_path),
                 static_cast<unsigned int>(iterations),
                 static_cast<unsigned int>(game_num),
                 static_cast<unsigned int>(mcts_num),
                 exploration_constant,
                 static_cast<unsigned int>(batch_size)};
  } else
    std::cerr << program;

  return std::monostate{};
}

} // namespace generals::cli
