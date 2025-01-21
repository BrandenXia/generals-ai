#!/usr/bin/env sh

if [ ! -d "build" ]; then
  mkdir build
fi

cmake -S . -B build -DCMAKE_PREFIX_PATH=$MAMBA_ROOT_PREFIX/envs/torch/lib
make -C build
