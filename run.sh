#!/bin/bash
set -e -x

mkdir -p build

CXX=clang++-17 CUDACXX=nvcc cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build

./build/hpps "$@"