#ifndef GENERALS_TRAIN_HPP
#define GENERALS_TRAIN_HPP

#include <filesystem>

namespace generals::train {

void interactive_train(std::filesystem::path network_path);

void train(int game_nums, int max_ticks, std::filesystem::path network_path);

} // namespace generals::train

#endif
