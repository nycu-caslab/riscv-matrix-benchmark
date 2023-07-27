#ifndef __SRC_MATMUL_H__
#define __SRC_MATMUL_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

//#define FP16_ACC16 1

static inline int matmul_rvv_uint8_m8(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    // assert(w1 == h2);

    // int hout = dst->h;
    // int wout = dst->w;

    // assert(hout == h1 && wout == w2);

    uint8_t *psrc1 = (uint8_t *)src1->data;
    uint8_t *psrc2 = (uint8_t *)src2->data;
    uint8_t *pdst = (uint8_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e8m8(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint8m8_t _sum;
            vuint8m8_t _zeros;
            // _sum = vsub_vv_u8m8(_zeros, _zeros, vl);
            asm volatile("vsub.vv %[vd], %[vs1], %[vs1]"
                            :[vd]"=vr"(_sum)
                            :[vs1]"vr"(_zeros));

            int offset_dst = i * w2 + j;
            uint8_t *_psrc1_off = psrc1 + i * w1;
            uint8_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint8_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint8m8_t _src2;
                // _src2 = vle8_v_u8m8(_psrc2_off, vl);
                asm volatile("vle8.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_src2)
                            :[rs1]"r"(_psrc2_off));
                // _sum = vmacc_vx_u8m8(_sum, _src1, _src2, vl);
                asm volatile("vmacc.vx %[vd], %[rs1], %[vs2]"
                            :[vd]"+vr"(_sum)
                            :[rs1]"r"(_src1), [vs2]"vr"(_src2));
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            // vse8_v_u8m8(pdst+offset_dst, _sum, vl);
            asm volatile("vse8.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_sum)
                            :[rs1]"r"(pdst+offset_dst));

        }
    }

    return 0;
}


static inline int matmul_rvv_uint8_m4(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    // assert(w1 == h2);

    // int hout = dst->h;
    // int wout = dst->w;

    // assert(hout == h1 && wout == w2);

    uint8_t *psrc1 = (uint8_t *)src1->data;
    uint8_t *psrc2 = (uint8_t *)src2->data;
    uint8_t *pdst = (uint8_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e8m4(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint8m4_t _sum;
            vuint8m4_t _zeros;
            // _sum = vsub_vv_u8m8(_zeros, _zeros, vl);
            asm volatile("vsub.vv %[vd], %[vs1], %[vs1]"
                            :[vd]"=vr"(_sum)
                            :[vs1]"vr"(_zeros));

            int offset_dst = i * w2 + j;
            uint8_t *_psrc1_off = psrc1 + i * w1;
            uint8_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint8_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint8m4_t _src2;
                // _src2 = vle8_v_u8m8(_psrc2_off, vl);
                asm volatile("vle8.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_src2)
                            :[rs1]"r"(_psrc2_off));
                // _sum = vmacc_vx_u8m8(_sum, _src1, _src2, vl);
                asm volatile("vmacc.vx %[vd], %[rs1], %[vs2]"
                            :[vd]"+vr"(_sum)
                            :[rs1]"r"(_src1), [vs2]"vr"(_src2));
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            // vse8_v_u8m8(pdst+offset_dst, _sum, vl);
            asm volatile("vse8.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_sum)
                            :[rs1]"r"(pdst+offset_dst));

        }
    }

    return 0;
}


static inline int matmul_rvv_uint8_m1(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    // assert(w1 == h2);

    // int hout = dst->h;
    // int wout = dst->w;

    // assert(hout == h1 && wout == w2);

    uint8_t *psrc1 = (uint8_t *)src1->data;
    uint8_t *psrc2 = (uint8_t *)src2->data;
    uint8_t *pdst = (uint8_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e8m1(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint8m1_t _sum;
            vuint8m1_t _zeros;
            // _sum = vsub_vv_u8m8(_zeros, _zeros, vl);
            asm volatile("vsub.vv %[vd], %[vs1], %[vs1]"
                            :[vd]"=vr"(_sum)
                            :[vs1]"vr"(_zeros));

            int offset_dst = i * w2 + j;
            uint8_t *_psrc1_off = psrc1 + i * w1;
            uint8_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint8_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint8m1_t _src2;
                // _src2 = vle8_v_u8m8(_psrc2_off, vl);
                asm volatile("vle8.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_src2)
                            :[rs1]"r"(_psrc2_off));
                // _sum = vmacc_vx_u8m8(_sum, _src1, _src2, vl);
                asm volatile("vmacc.vx %[vd], %[rs1], %[vs2]"
                            :[vd]"+vr"(_sum)
                            :[rs1]"r"(_src1), [vs2]"vr"(_src2));
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            // vse8_v_u8m8(pdst+offset_dst, _sum, vl);
            asm volatile("vse8.v %[vd], (%[rs1])"
                            :[vd]"=vr"(_sum)
                            :[rs1]"r"(pdst+offset_dst));

        }
    }

    return 0;
}

static inline int matmul_rvv_uint16_m8(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint16_t *psrc1 = (uint16_t *)src1->data;
    uint16_t *psrc2 = (uint16_t *)src2->data;
    uint16_t *pdst = (uint16_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e16m8(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint16m8_t _sum;
            vuint16m8_t _zeros;
            _sum = vsub_vv_u16m8(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint16_t *_psrc1_off = psrc1 + i * w1;
            uint16_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint16_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint16m8_t _src2;
                _src2 = vle16_v_u16m8(_psrc2_off, vl);
                _sum = vmacc_vx_u16m8(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse16_v_u16m8(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint16_m4(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint16_t *psrc1 = (uint16_t *)src1->data;
    uint16_t *psrc2 = (uint16_t *)src2->data;
    uint16_t *pdst = (uint16_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e16m4(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint16m4_t _sum;
            vuint16m4_t _zeros;
            _sum = vsub_vv_u16m4(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint16_t *_psrc1_off = psrc1 + i * w1;
            uint16_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint16_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint16m4_t _src2;
                _src2 = vle16_v_u16m4(_psrc2_off, vl);
                _sum = vmacc_vx_u16m4(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse16_v_u16m4(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint16_m1(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint16_t *psrc1 = (uint16_t *)src1->data;
    uint16_t *psrc2 = (uint16_t *)src2->data;
    uint16_t *pdst = (uint16_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e16m1(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint16m1_t _sum;
            vuint16m1_t _zeros;
            _sum = vsub_vv_u16m1(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint16_t *_psrc1_off = psrc1 + i * w1;
            uint16_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint16_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint16m1_t _src2;
                _src2 = vle16_v_u16m1(_psrc2_off, vl);
                _sum = vmacc_vx_u16m1(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse16_v_u16m1(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint32_m8(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint32_t *psrc1 = (uint32_t *)src1->data;
    uint32_t *psrc2 = (uint32_t *)src2->data;
    uint32_t *pdst = (uint32_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e32m8(w2 - j);

            printf("vl = %d\n", vl);
            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint32m8_t _sum;
            vuint32m8_t _zeros;
            _sum = vsub_vv_u32m8(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint32_t *_psrc1_off = psrc1 + i * w1;
            uint32_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint32_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint32m8_t _src2;
                _src2 = vle32_v_u32m8(_psrc2_off, vl);
                _sum = vmacc_vx_u32m8(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse32_v_u32m8(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint32_m4(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint32_t *psrc1 = (uint32_t *)src1->data;
    uint32_t *psrc2 = (uint32_t *)src2->data;
    uint32_t *pdst = (uint32_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e32m4(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint32m4_t _sum;
            vuint32m4_t _zeros;
            _sum = vsub_vv_u32m4(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint32_t *_psrc1_off = psrc1 + i * w1;
            uint32_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint32_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint32m4_t _src2;
                _src2 = vle32_v_u32m4(_psrc2_off, vl);
                _sum = vmacc_vx_u32m4(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse32_v_u32m4(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint32_m1(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint32_t *psrc1 = (uint32_t *)src1->data;
    uint32_t *psrc2 = (uint32_t *)src2->data;
    uint32_t *pdst = (uint32_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e32m1(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint32m1_t _sum;
            vuint32m1_t _zeros;
            _sum = vsub_vv_u32m1(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint32_t *_psrc1_off = psrc1 + i * w1;
            uint32_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint32_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint32m1_t _src2;
                _src2 = vle32_v_u32m1(_psrc2_off, vl);
                _sum = vmacc_vx_u32m1(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse32_v_u32m1(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint64_m8(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint64_t *psrc1 = (uint64_t *)src1->data;
    uint64_t *psrc2 = (uint64_t *)src2->data;
    uint64_t *pdst = (uint64_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e64m8(w2 - j);

            printf("vl = %d\n", vl);
            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint64m8_t _sum;
            vuint64m8_t _zeros;
            _sum = vsub_vv_u64m8(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint64_t *_psrc1_off = psrc1 + i * w1;
            uint64_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint64_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint64m8_t _src2;
                _src2 = vle64_v_u64m8(_psrc2_off, vl);
                _sum = vmacc_vx_u64m8(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse64_v_u64m8(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint64_m4(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint64_t *psrc1 = (uint64_t *)src1->data;
    uint64_t *psrc2 = (uint64_t *)src2->data;
    uint64_t *pdst = (uint64_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e64m4(w2 - j);
            printf("vl = %d\n", vl);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint64m4_t _sum;
            vuint64m4_t _zeros;
            _sum = vsub_vv_u64m4(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint64_t *_psrc1_off = psrc1 + i * w1;
            uint64_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint64_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint64m4_t _src2;
                _src2 = vle64_v_u64m4(_psrc2_off, vl);
                _sum = vmacc_vx_u64m4(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse64_v_u64m4(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}


static inline int matmul_rvv_uint64_m1(Tensor *dst, Tensor *src1, Tensor *src2)
{
    int h1 = src1->h;
    int w1 = src1->w;

    int h2 = src2->h;
    int w2 = src2->w;

    assert(w1 == h2);

    int hout = dst->h;
    int wout = dst->w;

    assert(hout == h1 && wout == w2);

    uint64_t *psrc1 = (uint64_t *)src1->data;
    uint64_t *psrc2 = (uint64_t *)src2->data;
    uint64_t *pdst = (uint64_t *)dst->data;

    int vl;
    for(int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j += vl) {
            vl = vsetvl_e64m1(w2 - j);

            // vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl);
            vuint64m1_t _sum;
            vuint64m1_t _zeros;
            _sum = vsub_vv_u64m1(_zeros, _zeros, vl);

            int offset_dst = i * w2 + j;
            uint64_t *_psrc1_off = psrc1 + i * w1;
            uint64_t *_psrc2_off = psrc2 + j;
            for (int k = 0; k < w1; k++) {
                uint64_t _src1 = *_psrc1_off;
                // vfloat16m4_t _src2 = vle16_v_f16m4(_psrc2_off, vl);
                // _sum = vfwmacc_vf_f32m8(_sum, _src1, _src2, vl);
                vuint64m1_t _src2;
                _src2 = vle64_v_u64m1(_psrc2_off, vl);
                _sum = vmacc_vx_u64m1(_sum, _src1, _src2, vl);
                
                _psrc1_off++;
                _psrc2_off += w2;
            }
            // vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m1(_sum, vl), vl);
            vse64_v_u64m1(pdst+offset_dst, _sum, vl);

        }
    }

    return 0;
}

#endif // __SRC_MATMUL_H__
