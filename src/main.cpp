#include <argparse/argparse.hpp>

#include "train.hpp"

using namespace generals;

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program{"generals-ai", "0.1.0"};
  program.add_description("generals.io neural-network-based AI bot");

  argparse::ArgumentParser train_command("train");
  train_command.add_description("Train the AI");
  train_command.add_argument("-g", "--games")
      .default_value(1000)
      .help("Number of games to train on")
      .scan<'d', int>();
  train_command.add_argument("-t", "--ticks")
      .default_value(1000)
      .help("Maximum number of ticks per game")
      .scan<'d', int>();

  argparse::ArgumentParser interactive_command("interactive");
  interactive_command.add_description("Interactive training");

  program.add_subparser(train_command);
  program.add_subparser(interactive_command);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  if (program.is_subcommand_used(train_command)) {
    auto game_nums = train_command.get<int>("--games");
    auto max_ticks = train_command.get<int>("--ticks");
    train::train(game_nums, max_ticks);
  } else if (program.is_subcommand_used(interactive_command))
    train::interactive_train();
  else
    std::cerr << program;

  return 0;
}
