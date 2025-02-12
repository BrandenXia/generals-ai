cmake_minimum_required(VERSION 3.30)
project(generals_ai)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined \
  -fno-omit-frame-pointer \
")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
include_directories(external/include)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(generals_ai ${SOURCES})

target_compile_definitions(generals_ai PRIVATE
  $<$<CONFIG:Debug>:DEBUG>
)

# libtorch
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
target_link_libraries(generals_ai PRIVATE "${TORCH_LIBRARIES}")

# raylib
find_package(raylib REQUIRED)
include_directories(${RAYLIB_INCLUDE_DIRS})
target_link_libraries(generals_ai PRIVATE raylib)
if (APPLE)
  target_link_libraries(generals_ai PRIVATE "-framework IOKit")
  target_link_libraries(generals_ai PRIVATE "-framework Cocoa")
  target_link_libraries(generals_ai PRIVATE "-framework OpenGL")
endif()

find_package(spdlog CONFIG REQUIRED)
include_directories(${SPDLOG_INCLUDE_DIRS})
target_link_libraries(generals_ai PRIVATE spdlog::spdlog)
