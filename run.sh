#!/bin/bash

set -e -x

mkdir -p build

CC=clang-17 CXX=clang++-17 cmake -S . -B build -DCMAKE_CUDA_COMPILER=clang++-17

cmake --build build

./build/hpps ${@}