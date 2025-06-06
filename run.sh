#!/bin/bash

set -e -x

mkdir -p build

CC=clang CXX=clang++ cmake -S . -B build

cmake --build build

./build/hpps ${@}