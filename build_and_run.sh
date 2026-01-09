#!/bin/bash

set -e

usage() {
    echo "Usage: $0 {validate|benchmark|profile} <kernel_file.cuh> Optional(<N>)"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

TARGET=$1
KERNEL_FILE=$2
shift 2

if [[ ! -f "$KERNEL_FILE" || "$KERNEL_FILE" != *.cuh ]]; then
    echo "Error: You must provide a valid kernel file ending in .cuh"
    usage
fi

CUDA_PATH=/usr/local/cuda cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DKERNEL_HEADER_FILENAME="$KERNEL_FILE"

case $TARGET in
    validate)
        cmake --build build --target validate
        ./build/validate "$@"
        ;;

    benchmark)
        cmake --build build --target benchmark
        ./build/benchmark "$@"
        ;;

    profile)
        cmake --build build --target benchmark
        sudo /opt/nvidia/nsight-compute/2025.1.1/ncu --log-file /dev/stdout --section MemoryWorkloadAnalysis --section SpeedOfLight --section PmSampling --print-summary per-kernel --target-processes all ./build/benchmark "$@"
        ;;

    *)
        usage
        ;;
esac

echo "Done." 
