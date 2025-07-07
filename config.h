#include <stdint.h>

constexpr uint32_t BLOCKDIM = 512;
constexpr uint64_t MAXBLOCKS = 1048576;

constexpr uint32_t WARPSIZE = 32;
constexpr uint64_t FULLMASK = 0xFFFFFFFF;