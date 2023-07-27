#ifndef __CONV_H__
#define __CONV_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>
#include "matmul.h"

#include "mme.h"

static inline int conv(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vlmax = VLENB * 4 / 2;

    float16_t *psrc = (float16_t *)src->data;
    // padding the input data
    if (pad_l + pad_r + pad_t + pad_b) {
        float16_t *psrcPad = (float16_t *)srcPad->data;
        psrcPad += pad_t * (win + pad_l + pad_r) * cin + pad_l * cin;  
        for (int i = 0; i < hin; i++) {
            for (int j = 0; j < win * cin; j += vlmax) {
                unsigned vl = min(vlmax, win * cin - j);
                vfloat16m4_t _data = vle16_v_f16m4(psrc, vl);
                psrc += vl;
                vse16_v_f16m4(psrcPad, _data, vl);
                psrcPad += vl;
            }
            psrcPad += (pad_r + pad_l) * cin;
        }
        psrc = (float16_t *)srcPad->data;
    }

    hin = hin + pad_t + pad_b;
    win = win + pad_l + pad_r;

    float16_t *pweight = (float16_t *)weight->data;
    float16_t *pdst = (float16_t *)dst->data;

    
    for (int i = 0; i < hout; i++) {
      for (int j = 0; j < wout; j++) { 
        for (int k = 0; k < cout; k += vlmax) { // complete  vlmax one time
          unsigned vl_out = min(vlmax, cout - k);
          int offset_dst = i * wout * cout + j * cout + k;
          vfloat32m8_t _sum = vfmv_v_f_f32m8(0.f, vl_out);

          int offset_src1 = i * stride_h * win * cin + j * stride_w * cin;
          for (int m = 0; m < kh; m++) {
            int offset_src = offset_src1;
            for (int n = 0; n < kw; n++) {
              int offset_weight = m * kw * cout * cin + n * cout * cin + k;
              float16_t *_psrc_off = psrc + offset_src;
              float16_t *_psrc_weight = pweight + offset_weight;
              for (int l = 0; l < cin; l++) {
                float16_t _src = *_psrc_off;
                _psrc_off++;
                vfloat16m4_t _weight = vle16_v_f16m4(_psrc_weight, vl_out);
                _psrc_weight += cout;
                _sum = vfwmacc_vf_f32m8(_sum, _src, _weight, vl_out);
              } // l
            offset_src += dilation_w * cin;
            } // n
          offset_src1 += dilation_h * win * cin;
          } // m

          vse16_v_f16m4(pdst+offset_dst, vfncvt_f_f_w_f16m4(_sum, vl_out), vl_out);
 
        } // k
      } // j
    } // i

    return 0;
}


static inline int conv_uint8(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{

    // printf("RVV");
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vlmax = VLENB * 4 / 2;

    uint8_t *psrc = (uint8_t *)src->data;
    // padding the input data
    if (pad_l + pad_r + pad_t + pad_b) {
        uint8_t *psrcPad = (uint8_t *)srcPad->data;
        psrcPad += pad_t * (win + pad_l + pad_r) * cin + pad_l * cin;  
        for (int i = 0; i < hin; i++) {
            for (int j = 0; j < win * cin; j += vlmax) {
                unsigned vl = min(vlmax, win * cin - j);
                vuint8m1_t _data = vle8_v_u8m1(psrc, vl);
                psrc += vl;
                vse8_v_u8m1(psrcPad, _data, vl);
                psrcPad += vl;
            }
            psrcPad += (pad_r + pad_l) * cin;
        }
        psrc = (uint8_t *)srcPad->data;
    }

    hin = hin + pad_t + pad_b;
    win = win + pad_l + pad_r;

    uint8_t *pweight = (uint8_t *)weight->data;
    uint8_t *pdst = (uint8_t *)dst->data;

    
    for (int i = 0; i < hout; i++) {
      for (int j = 0; j < wout; j++) { 
        for (int k = 0; k < cout; k += vlmax) { // complete  vlmax one time
          unsigned vl_out = min(vlmax, cout - k);
          int offset_dst = i * wout * cout + j * cout + k;
          vuint8m1_t _sum = vmv_v_x_u8m1(0, vl_out);

          int offset_src1 = i * stride_h * win * cin + j * stride_w * cin;
          for (int m = 0; m < kh; m++) {
            int offset_src = offset_src1;
            for (int n = 0; n < kw; n++) {
              int offset_weight = m * kw * cout * cin + n * cout * cin + k;
              uint8_t *_psrc_off = psrc + offset_src;
              uint8_t *_psrc_weight = pweight + offset_weight;
              for (int l = 0; l < cin; l++) {
                uint8_t _src = *_psrc_off;
                _psrc_off++;
                vuint8m1_t _weight = vle8_v_u8m1(_psrc_weight, vl_out);
                _psrc_weight += cout;
                _sum = vmacc_vx_u8m1(_sum, _src, _weight, vl_out);
              } // l
            offset_src += dilation_w * cin;
            } // n
          offset_src1 += dilation_h * win * cin;
          } // m

          vse8_v_u8m1(pdst+offset_dst, _sum, vl_out);
 
        } // k
      } // j
    } // i

    return 0;
}

static inline int conv_uint16(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{

    // printf("RVV");
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vlmax = VLENB * 4 / 2;

    uint16_t *psrc = (uint16_t *)src->data;
    // padding the input data
    if (pad_l + pad_r + pad_t + pad_b) {
        uint16_t *psrcPad = (uint16_t *)srcPad->data;
        psrcPad += pad_t * (win + pad_l + pad_r) * cin + pad_l * cin;  
        for (int i = 0; i < hin; i++) {
            for (int j = 0; j < win * cin; j += vlmax) {
                unsigned vl = min(vlmax, win * cin - j);
                vuint16m1_t _data = vle16_v_u16m1(psrc, vl);
                psrc += vl;
                vse16_v_u16m1(psrcPad, _data, vl);
                psrcPad += vl;
            }
            psrcPad += (pad_r + pad_l) * cin;
        }
        psrc = (uint16_t *)srcPad->data;
    }

    hin = hin + pad_t + pad_b;
    win = win + pad_l + pad_r;

    uint16_t *pweight = (uint16_t *)weight->data;
    uint16_t *pdst = (uint16_t *)dst->data;

    
    for (int i = 0; i < hout; i++) {
      for (int j = 0; j < wout; j++) { 
        for (int k = 0; k < cout; k += vlmax) { // complete  vlmax one time
          unsigned vl_out = min(vlmax, cout - k);
          int offset_dst = i * wout * cout + j * cout + k;
          vuint16m1_t _sum = vmv_v_x_u16m1(0, vl_out);

          int offset_src1 = i * stride_h * win * cin + j * stride_w * cin;
          for (int m = 0; m < kh; m++) {
            int offset_src = offset_src1;
            for (int n = 0; n < kw; n++) {
              int offset_weight = m * kw * cout * cin + n * cout * cin + k;
              uint16_t *_psrc_off = psrc + offset_src;
              uint16_t *_psrc_weight = pweight + offset_weight;
              for (int l = 0; l < cin; l++) {
                uint16_t _src = *_psrc_off;
                _psrc_off++;
                vuint16m1_t _weight = vle16_v_u16m1(_psrc_weight, vl_out);
                _psrc_weight += cout;
                _sum = vmacc_vx_u16m1(_sum, _src, _weight, vl_out);
              } // l
            offset_src += dilation_w * cin;
            } // n
          offset_src1 += dilation_h * win * cin;
          } // m

          vse16_v_u16m1(pdst+offset_dst, _sum, vl_out);
 
        } // k
      } // j
    } // i

    return 0;
}


static inline int conv_uint32(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{

    // printf("RVV");
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vlmax = VLENB * 4 / 2;

    uint32_t *psrc = (uint32_t *)src->data;
    // padding the input data
    if (pad_l + pad_r + pad_t + pad_b) {
        uint32_t *psrcPad = (uint32_t *)srcPad->data;
        psrcPad += pad_t * (win + pad_l + pad_r) * cin + pad_l * cin;  
        for (int i = 0; i < hin; i++) {
            for (int j = 0; j < win * cin; j += vlmax) {
                unsigned vl = min(vlmax, win * cin - j);
                vuint32m1_t _data = vle32_v_u32m1(psrc, vl);
                psrc += vl;
                vse32_v_u32m1(psrcPad, _data, vl);
                psrcPad += vl;
            }
            psrcPad += (pad_r + pad_l) * cin;
        }
        psrc = (uint32_t *)srcPad->data;
    }

    hin = hin + pad_t + pad_b;
    win = win + pad_l + pad_r;

    uint32_t *pweight = (uint32_t *)weight->data;
    uint32_t *pdst = (uint32_t *)dst->data;

    
    for (int i = 0; i < hout; i++) {
      for (int j = 0; j < wout; j++) { 
        for (int k = 0; k < cout; k += vlmax) { // complete  vlmax one time
          unsigned vl_out = min(vlmax, cout - k);
          int offset_dst = i * wout * cout + j * cout + k;
          vuint32m1_t _sum = vmv_v_x_u32m1(0, vl_out);

          int offset_src1 = i * stride_h * win * cin + j * stride_w * cin;
          for (int m = 0; m < kh; m++) {
            int offset_src = offset_src1;
            for (int n = 0; n < kw; n++) {
              int offset_weight = m * kw * cout * cin + n * cout * cin + k;
              uint32_t *_psrc_off = psrc + offset_src;
              uint32_t *_psrc_weight = pweight + offset_weight;
              for (int l = 0; l < cin; l++) {
                uint32_t _src = *_psrc_off;
                _psrc_off++;
                vuint32m1_t _weight = vle32_v_u32m1(_psrc_weight, vl_out);
                _psrc_weight += cout;
                _sum = vmacc_vx_u32m1(_sum, _src, _weight, vl_out);
              } // l
            offset_src += dilation_w * cin;
            } // n
          offset_src1 += dilation_h * win * cin;
          } // m

          vse32_v_u32m1(pdst+offset_dst, _sum, vl_out);
 
        } // k
      } // j
    } // i

    return 0;
}


static inline int conv_uint64(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{

    // printf("RVV");
    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vlmax = VLENB * 4 / 2;

    uint64_t *psrc = (uint64_t *)src->data;
    // padding the input data
    if (pad_l + pad_r + pad_t + pad_b) {
        uint64_t *psrcPad = (uint64_t *)srcPad->data;
        psrcPad += pad_t * (win + pad_l + pad_r) * cin + pad_l * cin;  
        for (int i = 0; i < hin; i++) {
            for (int j = 0; j < win * cin; j += vlmax) {
                unsigned vl = min(vlmax, win * cin - j);
                vuint64m1_t _data = vle64_v_u64m1(psrc, vl);
                psrc += vl;
                vse64_v_u64m1(psrcPad, _data, vl);
                psrcPad += vl;
            }
            psrcPad += (pad_r + pad_l) * cin;
        }
        psrc = (uint64_t *)srcPad->data;
    }

    hin = hin + pad_t + pad_b;
    win = win + pad_l + pad_r;

    uint64_t *pweight = (uint64_t *)weight->data;
    uint64_t *pdst = (uint64_t *)dst->data;

    
    for (int i = 0; i < hout; i++) {
      for (int j = 0; j < wout; j++) { 
        for (int k = 0; k < cout; k += vlmax) { // complete  vlmax one time
          unsigned vl_out = min(vlmax, cout - k);
          int offset_dst = i * wout * cout + j * cout + k;
          vuint64m1_t _sum = vmv_v_x_u64m1(0, vl_out);

          int offset_src1 = i * stride_h * win * cin + j * stride_w * cin;
          for (int m = 0; m < kh; m++) {
            int offset_src = offset_src1;
            for (int n = 0; n < kw; n++) {
              int offset_weight = m * kw * cout * cin + n * cout * cin + k;
              uint64_t *_psrc_off = psrc + offset_src;
              uint64_t *_psrc_weight = pweight + offset_weight;
              for (int l = 0; l < cin; l++) {
                uint64_t _src = *_psrc_off;
                _psrc_off++;
                vuint64m1_t _weight = vle64_v_u64m1(_psrc_weight, vl_out);
                _psrc_weight += cout;
                _sum = vmacc_vx_u64m1(_sum, _src, _weight, vl_out);
              } // l
            offset_src += dilation_w * cin;
            } // n
          offset_src1 += dilation_h * win * cin;
          } // m

          vse64_v_u64m1(pdst+offset_dst, _sum, vl_out);
 
        } // k
      } // j
    } // i

    return 0;
}

static inline int conv_rvv_uint8(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{

    int stride_h = ss->stride_h;
    int stride_w = ss->stride_w;

    int pad_t = ss->top;
    int pad_b = ss->bottom;
    int pad_l = ss->left;
    int pad_r = ss->right;

    int dilation_h = ss->dilation_h;
    int dilation_w = ss->dilation_w;

    int kh = ss->kh;
    int kw = ss->kw;

    int hin = ss->hin;
    int win = ss->win;
    int cin = ss->cin;

    int hout = ss->hout;
    int wout = ss->wout;
    int cout = ss->cout;

    int vlmax = VLENB * 4 / 2;

    uint8_t *psrc = (uint8_t *)src->data;
    uint8_t *psrcPad = (uint8_t *)srcPad->data;
    memset(psrcPad, 0, hout * wout * kh * kw * cin * sizeof(uint8_t));
    tensor_new_2d(weight_2d, kh*kw*cin, cout, weight->elemsize, weight->data);
    tensor_new_2d(dst_2d, hout*wout, cout, dst->elemsize, dst->data);

    // im2col
    for (int i = 0; i < hout; i++) {
      for (int j = 0; j < wout; j++) {

        for (int ii = 0; ii < kh; ii++) {
          for (int jj = 0; jj < kw; jj++) {
            int _i = i * stride_h - pad_t + ii * dilation_h;
            int _j = j * stride_w - pad_l + jj * dilation_w;
            if (_i < 0 || _i > hin - 1) continue;
            if (_j < 0 || _j > win - 1) continue;
            for (int kk = 0; kk < cin; kk++) {
              int xpos = i * wout * kh * kw * cin + j * kh * kw * cin + ii * kw * cin + jj * cin + kk;
              int spos = _i * win * cin + _j * cin + kk;
              psrcPad[xpos] = psrc[spos];
            }
          }
        }
        
      }
    }
    // printf("n m k : %d %d %d\n", weight_2d.w, srcPad->h, srcPad->w)
    
    matmul_rvv_uint8(&dst_2d, srcPad, &weight_2d);

    return 0;
}




#endif