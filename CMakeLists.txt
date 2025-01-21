cmake_minimum_required(VERSION 3.30)
project(generals_ai)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer"
)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(generals_ai ${SOURCES})

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
target_link_libraries(generals_ai PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET generals_ai PROPERTY CXX_STANDARD 23)
