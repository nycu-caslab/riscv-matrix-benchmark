#ifndef __SRC_MATMUL_INT_SCALAR_HH__
#define __SRC_MATMUL_INT_SCALAR_HH__


#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "../include/matrix/matrix_intrinsic.h"


static inline int matmul_scalar_uint8(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h; //m
    int w1 = src1->w; //k

    int h2 = src2->h;
    int w2 = src2->w; //n

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint8_t *psrc1 = (uint8_t *)src1->data; //ap
    uint8_t *psrc2 = (uint8_t *)src2->data; //bp
    uint8_t *pdst = (uint8_t *)dst->data;   //cp

    int i=0, j=0, k=0;
    for(i=0; i< h1; i++) {
      for(k=0;k<w1;k++) {
        for(j=0;j<w2;j++) {
          pdst[i*w2+j] += psrc1[i*w1+k] * psrc2[k*w2+j];
        }
      }
    }

    return 0;
}


static inline int matmul_scalar_uint16(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h; //m
    int w1 = src1->w; //k

    int h2 = src2->h;
    int w2 = src2->w; //n

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint16_t *psrc1 = (uint16_t *)src1->data; //ap
    uint16_t *psrc2 = (uint16_t *)src2->data; //bp
    uint16_t *pdst = (uint16_t *)dst->data;   //cp

    const int dataSize = sizeof(uint16_t);

    int i=0, j=0,k=0;
    for(i=0; i< h1; i++) {
      for(k=0;k<w1;k++) {
        for(j=0;j<w2;j++) {
          pdst[i*w2+j] += psrc1[i*w1+k] * psrc2[k*w2+j];
        }
      }
    }

    return 0;
}


static inline int matmul_scalar_uint32(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h; //m
    int w1 = src1->w; //k

    register int h2 = src2->h;
    int w2 = src2->w; //n

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint32_t *psrc1 = (uint32_t *)src1->data; //ap
    uint32_t *psrc2 = (uint32_t *)src2->data; //bp
    uint32_t *pdst = (uint32_t *)dst->data;   //cp

    int i=0, j=0,k=0;
    for(i=0; i<h1; i++) {
      for(k=0;k<w1;k++) {
        for(j=0;j<w2;j++) {
          pdst[i*w2+j] += psrc1[i*w1+k] * psrc2[k*w2+j];
        }
      }
    }
    

    return 0;
}


static inline int matmul_scalar_uint64(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h; //m
    int w1 = src1->w; //k

    int h2 = src2->h;
    int w2 = src2->w; //n

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint64_t *psrc1 = (uint64_t *)src1->data; //ap
    uint64_t *psrc2 = (uint64_t *)src2->data; //bp
    uint64_t *pdst = (uint64_t *)dst->data;   //cp

    int i=0, j=0,k=0;
    for(i=0; i<h1; i++) {
      for(k=0;k<w1;k++) {
        for(j=0;j<w2;j++) {
          pdst[i*w2+j] += psrc1[i*w1+k] * psrc2[k*w2+j];
        }
      }
    }
   
    return 0;
}



#endif