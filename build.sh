#!/bin/bash
set -e -x

mkdir -p build

CXX=clang++-17 cmake -S . -B build -DCMAKE_CUDA_COMPILER=clang++-17
cmake --build build

./build/hpps "$@"


ncu \
  --set detailed \
  --print-summary per-kernel \
  -k single_pass_scan \
  --log-file roofline_report.log \
  ./build/hpps 24