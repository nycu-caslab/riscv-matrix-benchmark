#ifndef __CONV_H__
#define __CONV_H__

#include "tensor.h"
#include <stddef.h>
#include <riscv_vector.h>

#include "mme.h"
#include "matmul.h"

static inline int conv_matrix(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
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
    float16_t *psrcPad = (float16_t *)srcPad->data;
    memset(psrcPad, 0, hout * wout * kh * kw * cin * sizeof(float16_t));
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
    
    return matmul(&dst_2d, srcPad, &weight_2d);
}


static inline int conv_rvm_uint8(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
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
    printf("n m k : %d %d %d\n", weight_2d.w, srcPad->h, srcPad->w);
    // matmul_rvm_uint8(&dst_2d, srcPad, &weight_2d);
    return 0;
}


static inline int conv_rvm_uint8_indir_buffer(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Tensor *bufPad, Config *ss)
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
    uint8_t**psrcPad = (uint8_t **)srcPad->data;
    //memset(psrcPad, 0, hout * wout * kh * kw * sizeof(uint8_t*));
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
            int xpos = i * wout * kh * kw + j * kh * kw+ ii * kw + jj;
            int spos = _i * win * cin + _j * cin;
            psrcPad[xpos] = &psrc[spos];
          }
        }
      }
    }
    // printf("n m k : %d %d %d\n", weight_2d.w, srcPad->h, srcPad->w);
    matmul_rvm_uint8_indir_buffer(&dst_2d, srcPad, &weight_2d, ss->cin, bufPad);
    return 0;
}


static inline int conv_rvm_uint16(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
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

    uint16_t *psrc = (uint16_t *)src->data;
    uint16_t *psrcPad = (uint16_t *)srcPad->data;
    memset(psrcPad, 0, hout * wout * kh * kw * cin * sizeof(uint16_t));
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
    // printf("n m k : %d %d %d\n", weight_2d.w, srcPad->h, srcPad->w);
    matmul_rvm_uint16(&dst_2d, srcPad, &weight_2d);
    return 0;
}



static inline int conv_rvm_uint32(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
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

    uint32_t *psrc = (uint32_t *)src->data;
    uint32_t *psrcPad = (uint32_t *)srcPad->data;
    memset(psrcPad, 0, hout * wout * kh * kw * cin * sizeof(uint32_t));
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
    // printf("n m k : %d %d %d\n", weight_2d.w, srcPad->h, srcPad->w);
    matmul_rvm_uint32(&dst_2d, srcPad, &weight_2d);
    return 0;
}


static inline int conv_rvm_uint64(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
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

    uint64_t *psrc = (uint64_t *)src->data;
    uint64_t *psrcPad = (uint64_t *)srcPad->data;
    memset(psrcPad, 0, hout * wout * kh * kw * cin * sizeof(uint64_t));
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
    // printf("n m k : %d %d %d\n", weight_2d.w, srcPad->h, srcPad->w);
    matmul_rvm_uint64(&dst_2d, srcPad, &weight_2d);
    return 0;
}


static inline int conv(Tensor *dst, Tensor *src, Tensor *weight, Tensor *srcPad, Config *ss)
{
  conv_matrix(dst, src, weight, srcPad, ss);
  return 0;
}

#endif
