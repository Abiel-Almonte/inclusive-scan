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

cmake -S . -B build -DKERNEL_HEADER_FILENAME="$KERNEL_FILE"

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
        sudo /usr/local/NVIDIA-Nsight-Compute/ncu --log-file /dev/stdout --section MemoryWorkloadAnalysis --section SpeedOfLight --section PmSampling --pm-sampling-buffer-size 134217728 --pm-sampling-interval 10000 --pm-sampling-max-passes 25 --print-summary per-kernel --target-processes all ./build/benchmark "$@"
        ;;

    *)
        usage
        ;;
esac

echo "Done." 
