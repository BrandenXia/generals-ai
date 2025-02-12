CMAKE_BUILD_COMMAND := cmake -S . -B build -DCMAKE_PREFIX_PATH=$(MAMBA_ROOT_PREFIX)/envs/torch/lib
CMAKE_MAKE_COMMAND := cmake --build build

ifndef MAMBA_ROOT_PREFIX
	$(error MAMBA_ROOT_PREFIX is not set)
endif


.PHONY: all debug release run clean

all: debug

debug: build
	$(CMAKE_BUILD_COMMAND) -DCMAKE_BUILD_TYPE=Debug
	$(CMAKE_MAKE_COMMAND)

release: build
	$(CMAKE_BUILD_COMMAND) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE_MAKE_COMMAND)

run:
	./build/generals_ai

build:
	mkdir -p build

clean:
	rm -rf build
