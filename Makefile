CMAKE_BUILD_COMMAND := cmake -S . -B build
CMAKE_MAKE_COMMAND := cmake --build build

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
