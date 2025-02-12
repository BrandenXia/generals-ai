#include <spdlog/spdlog.h>

#include "train.hpp"

using namespace generals;

int main() {
#ifdef DEBUG
  spdlog::set_level(spdlog::level::debug);
#endif
  train::interactive_train();
}
