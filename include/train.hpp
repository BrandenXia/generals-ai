#ifndef GENERALS_TRAIN_HPP
#define GENERALS_TRAIN_HPP

#include <filesystem>

namespace generals::train {

void interactive_train(const std::filesystem::path &network_path);

void train(int game_nums, int max_ticks,
           const std::filesystem::path &network_path);

void bidirectional_train(int game_nums, int max_ticks,
                         const std::filesystem::path &n1_path,
                         const std::filesystem::path &n2_path);

} // namespace generals::train

#endif
