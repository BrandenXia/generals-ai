#ifndef GENERALS_DIRS_HPP
#define GENERALS_DIRS_HPP

#include <filesystem>

inline const std::filesystem::path ROOT_DIR = std::filesystem::current_path();
inline const std::filesystem::path DATA_DIR = ROOT_DIR / "data";

#endif
