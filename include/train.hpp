#ifndef GENERALS_TRAIN_HPP
#define GENERALS_TRAIN_HPP

#include <filesystem>

#include "game.hpp"

namespace generals::train {

void interactive_train(std::filesystem::path network_path,
                       game::Player interact_player, game::Player opponent);

void train(int game_nums, int max_ticks, std::filesystem::path network_path,
           game::Player player);

void bidirectional_train(int game_nums, int max_ticks,
                         std::filesystem::path n1_path,
                         std::filesystem::path n2_path, game::Player n1_player,
                         game::Player n2_player);

} // namespace generals::train

#endif
