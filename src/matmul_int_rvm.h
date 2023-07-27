#ifndef __SRC_MATMUL_INT_HH__
#define __SRC_MATMUL_INT_HH__


#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "../include/matrix/matrix_intrinsic.h"


static inline int matmul_rvm_uint8(Tensor *dst, Tensor *src1, Tensor *src2)
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

    const int dataSize = sizeof(uint8_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e8; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    int i=0, j=0, k=0;
    for(i=0; i < h1; i += tile_m){
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
      for(j=0; j< w2; j += tile_n) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tile_n)
                    : [rs1]"r"(w2-j));
        asm volatile("mwemulc.mi acc0, acc0, 0");
        for(k = 0; k < w1; k+=tile_k1) {
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          asm volatile("mlae8.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe8.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));

          asm volatile("mma.mm acc0, tr0, tr1");
        }

        asm volatile("msce8.m acc0, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
      }
    }

    return 0;
}


static inline int matmul_rvm_uint8_unrolling(Tensor *dst, Tensor *src1, Tensor *src2)
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

    const int dataSize = sizeof(uint8_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e8; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    int i=0, j=0, k=0;
    int step = 0;
    int step_k = 0;
    for(i=0; i < h1; i += tile_m){
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
      for(j=0; j< w2; j += tile_n) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tile_n)
                    : [rs1]"r"(w2-j));
        asm volatile("mwemulc.mi acc0, acc0, 0");
        asm volatile("mwemulc.mi acc0, acc0, 0");
        step = w1 / (4*2);
        for(step_k=0,k=0;step_k<step;step_k++) {
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          asm volatile("mlae8.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe8.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));
          k += tile_k1;
          asm volatile("mma.mm acc0, tr0, tr1");

          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          asm volatile("mlae8.m tr2, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe8.m tr3, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));
          k += tile_k1;
          asm volatile("mma.mm acc1, tr2, tr3");
        }

        asm volatile("msce8.m acc0, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
      }
    }

    return 0;
}





static inline int matmul_rvm_uint8_indir_buffer(Tensor *dst, Tensor *src1, Tensor *src2, int cin, Tensor* bufPad)
{
    int h1 = src1->h; //m
    int w1 = src1->w; //k

    int h2 = src2->h;
    int w2 = src2->w; //n

    int hout = dst->h;
    int wout = dst->w;
    int bufw = bufPad->w;

    uint8_t **psrc1 = (uint8_t **)src1->data; //ap
    uint8_t *psrc2 = (uint8_t *)src2->data; //bp
    uint8_t *pdst = (uint8_t *)dst->data;   //cp
    uint8_t *buf = (uint8_t*)bufPad->data; // 4 * 16

    const int dataSize = sizeof(uint8_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e8; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    int i=0, j=0, k=0, kk=0;
    int bi=0;
    for(i=0; i < h1; i += tile_m){
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
      for(j=0; j< w2; j += tile_n) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tile_n)
                    : [rs1]"r"(w2-j));
        asm volatile("mwemulc.mi acc0, acc0, 0");
        for(kk = 0; kk < w1; kk++) { //* loop for the indirect buffer
          for(k = 0;k < cin; k+=tile_k1) {
            asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(cin-k));
            for(bi=0; bi < tile_m; bi++) {
              uint8_t* ptr = *(psrc1 + (i+bi) * w1) + k; 
              memcpy(buf+bi*bufPad->w, ptr, tile_k1 * dataSize);
            }
            asm volatile("mlae8.m tr0, (%[rs1]), %[rs2]"
                              :
                              :[rs1]"r"(buf), [rs2]"r"(bufw*dataSize));
            asm volatile("mlbe8.m tr1, (%[rs1]), %[rs2]"
                        :
                        :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));

            asm volatile("mma.mm acc0, tr0, tr1");
          }
        }
        asm volatile("msce8.m acc0, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
      }
    }

    return 0;
}
static inline int matmul_rvm_uint16(Tensor *dst, Tensor *src1, Tensor *src2)
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

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e16; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    int i=0, j=0, k=0;
    for(i=0; i < h1; i += tile_m){
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
      for(j=0; j< w2; j += tile_n) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tile_n)
                    : [rs1]"r"(w2-j));
        asm volatile("mwemulc.mi acc0, acc1, 0");
        for(k = 0; k < w1; k+=tile_k1) {
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          asm volatile("mlae16.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe16.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));

          asm volatile("mma.mm acc0, tr0, tr1");
        }

        asm volatile("msce16.m acc0, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
      }
    }

    return 0;
}


static inline int matmul_rvm_uint32(Tensor *dst, Tensor *src1, Tensor *src2)
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

    const int dataSize = sizeof(uint32_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e32; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    int i=0, j=0, k=0;
    for(i=0; i < h1; i += tile_m){
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
      for(j=0; j< w2; j += tile_n) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tile_n)
                    : [rs1]"r"(w2-j));
        asm volatile("mwemulc.mi acc0, acc1, 0");
        for(k = 0; k < w1; k+=tile_k1) {
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          asm volatile("mlae32.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe32.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));

          asm volatile("mma.mm acc0, tr0, tr1");
        }

        asm volatile("msce32.m acc0, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
      }
    }

    return 0;
}


static inline int matmul_rvm_uint32_opt(Tensor *dst, Tensor *src1, Tensor *src2)
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

    const int dataSize = sizeof(uint32_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e32; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    int i=0, j=0, k=0;
    for(i=0; i < h1; i += tile_m){
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
      for(j=0; j< w2; j += tile_n) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tile_n)
                    : [rs1]"r"(w2-j));
        asm volatile("mwemulc.mi acc0, acc0, 0");
        int k_4group = w1 / 4;
        int k_step = 0;
        for(k = 0; k < k_4group; k+=k_step) {
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          asm volatile("mlae32.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe32.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));

          asm volatile("mma.mm acc0, tr0, tr1");
          k_step+=tile_k1;
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          
          asm volatile("mlae32.m tr2, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe32.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));
          asm volatile("mma.mm acc0, tr0, tr1");
          k_step+=tile_k1;
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          
          asm volatile("mlae32.m tr2, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe32.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));
          asm volatile("mma.mm acc0, tr0, tr1");
          k_step+=tile_k1;
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          
          asm volatile("mlae32.m tr2, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe32.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));
          asm volatile("mma.mm acc0, tr0, tr1");
        }

        asm volatile("msce32.m acc0, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
      }
    }

    return 0;
}


static inline int matmul_rvm_uint64(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h; //m
    int w1 = src1->w; //k

    // register int h2 = src2->h;
    int w2 = src2->w; //n

    // assert(w1 == h2);

    // int hout = dst->h;
    // int wout = dst->w;

    // assert(hout == h1 && wout == w2);

    uint64_t *psrc1 = (uint64_t *)src1->data; //ap
    uint64_t *psrc2 = (uint64_t *)src2->data; //bp
    uint64_t *pdst = (uint64_t *)dst->data;   //cp

    const int dataSize = sizeof(uint64_t);

    int tile_m = 0, tile_n = 0, tile_k1 = 0, tile_k2 = 0;
    int mtype = e64; // sew = e16, mlmul = 128
    asm volatile("msettype x0, %[rs1]"
                : 
                : [rs1]"r"(mtype));
    int i=0, j=0, k=0;
    for(i=0; i < h1; i += tile_m){
      asm volatile("msettilem %[rd], %[rs1]"
                    : [rd]"=r"(tile_m)
                    : [rs1]"r"(h1-i));
      for(j=0; j< w2; j += tile_n) {
        asm volatile("msettilen %[rd], %[rs1]"
                    : [rd]"=r"(tile_n)
                    : [rs1]"r"(w2-j));
        asm volatile("mwemulc.mi acc0, acc1, 0");
        for(k = 0; k < w1; k+=tile_k1) {
          asm volatile("msettilek %[rd], %[rs1]"
                      : [rd]"=r"(tile_k1)
                      : [rs1]"r"(w1-k));
          asm volatile("mlae64.m tr0, (%[rs1]), %[rs2]"
                            :
                            :[rs1]"r"(psrc1+i*w1+k), [rs2]"r"(w1*dataSize));
          asm volatile("mlbe64.m tr1, (%[rs1]), %[rs2]"
                      :
                      :[rs1]"r"(psrc2+k*w2+j), [rs2]"r"(w2*dataSize));

          asm volatile("mma.mm acc0, tr0, tr1");
        }

        asm volatile("msce64.m acc0, (%[rs1]), %[rs2]"
                      : 
                      : [rs1]"r"(pdst+i*w2+j), [rs2]"r"(w2*dataSize));
      }
    }

    return 0;
}



#endif