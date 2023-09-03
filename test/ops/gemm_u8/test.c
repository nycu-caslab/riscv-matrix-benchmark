#include <stdio.h>
#include <stdlib.h>

#include "../../../src/matmul.h"
#include "../../../src/perf.h"
#include "../../../include/incbin.h"

#include "params.h"

INCBIN(src1Data, "src1.bin", ".scdata1.params");
INCBIN(src2Data, "src2.bin", ".scdata2.params");

uint8_t dstData[OUT_SIZE * sizeof(uint8_t)] __attribute__((__section__(".scdata.output")));

int main(int argc, char **argv)
{
    const int m = M;
    const int k = K;
    const int n = N;

    tensor_new_2d(src1Mat, m, k, sizeof(uint8_t), src1Data);
    tensor_new_2d(src2Mat, k, n, sizeof(uint8_t), src2Data);
    tensor_new_2d(dstMat, m, n, sizeof(uint8_t), dstData);
    // matmul_rvm_uint8(&dstMat, &src1Mat, &src2Mat);
    matmul_rvm_uint8_two_level_tiling(&dstMat, &src1Mat, &src2Mat);

    return 0;
}
