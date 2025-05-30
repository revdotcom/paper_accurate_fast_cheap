#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef float fp32;
typedef double fp64;
void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, bf16 *w, bf16 *u, bf16 *y);
void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, bf16 *w, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<bf16>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<bf16>(), u.data_ptr<bf16>(), gy.data_ptr<bf16>(), gr.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>());
}
void cuda_forward_fp32(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *y);
void cuda_backward_fp32(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *gy, fp32 *gr, fp32 *gk, fp32 *gv, fp32 *gw, fp32 *gu);

void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp32(B, T, C, H, r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<fp32>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}
void backward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_fp32(B, T, C, H, r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<fp32>(), u.data_ptr<fp32>(), gy.data_ptr<fp32>(), gr.data_ptr<fp32>(), gk.data_ptr<fp32>(), gv.data_ptr<fp32>(), gw.data_ptr<fp32>(), gu.data_ptr<fp32>());
}
void cuda_forward_fp64(int B, int T, int C, int H, fp64 *r, fp64 *k, fp64 *v, fp64 *w, fp64 *u, fp64 *y);
void cuda_backward_fp64(int B, int T, int C, int H, fp64 *r, fp64 *k, fp64 *v, fp64 *w, fp64 *u, fp64 *gy, fp64 *gr, fp64 *gk, fp64 *gv, fp64 *gw, fp64 *gu);

void forward_fp64(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp64(B, T, C, H, r.data_ptr<fp64>(), k.data_ptr<fp64>(), v.data_ptr<fp64>(), w.data_ptr<fp64>(), u.data_ptr<fp64>(), y.data_ptr<fp64>());
}
void backward_fp64(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward_fp64(B, T, C, H, r.data_ptr<fp64>(), k.data_ptr<fp64>(), v.data_ptr<fp64>(), w.data_ptr<fp64>(), u.data_ptr<fp64>(), gy.data_ptr<fp64>(), gr.data_ptr<fp64>(), gk.data_ptr<fp64>(), gv.data_ptr<fp64>(), gw.data_ptr<fp64>(), gu.data_ptr<fp64>());
}

TORCH_LIBRARY(wkv6, m) {
    m.def("forward", forward);
    m.def("backward", backward);
    m.def("forward_fp32", forward_fp32);
    m.def("backward_fp32", backward_fp32);
    m.def("forward_fp64", forward_fp64);
    m.def("backward_fp64", backward_fp64);
}