// Backward kernel for assemble_fused. See docs/cuda_architecture.md.
//
// One thread per world-Gaussian. Recomputes forward intermediates (cheap)
// so it can route upstream grads (d_wm, d_ws, d_wr, d_wo, d_wf) back to
// the splat-side (d_lm, d_ls, d_lq, d_lo, d_lf) and the instance-side
// (d_ip, d_ir, d_iscl, d_iopa) via atomicAdd.
//
// Both splat and instance scatter use plain atomicAdd. Consecutive `g`s
// tend to share inst_idx because inst_offsets is a prefix sum, so the
// hardware warp atomic coalescer already collapses many of these.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "Common.cuh"

namespace scion_cuda {

// Binary-search inst_offsets[] for the inst_idx containing g.
__device__ __forceinline__ int find_inst_idx_bwd(
    const int32_t* __restrict__ inst_offsets,
    int N, int g
) {
    int lo = 0, hi = N;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (inst_offsets[mid + 1] <= g) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void assemble_fused_bwd_kernel(
    // Forward inputs (read-only).
    const float* __restrict__ splat_means,
    const float* __restrict__ splat_scales,
    const float* __restrict__ splat_quats,
    const float* __restrict__ splat_opa,           // for wo-clamp mask
    /* splat_feat_dc not needed (d_lf = d_wf directly) */
    const float* __restrict__ inst_rot,
    const float* __restrict__ inst_scl,
    const float* __restrict__ inst_opa,            // for wo-clamp mask
    const int32_t* __restrict__ cand_splat_rows,
    const int32_t* __restrict__ inst_offsets,
    // Upstream grads
    const float* __restrict__ d_world_means,       // [total, 3]
    const float* __restrict__ d_world_scales,      // [total, 3]
    const float* __restrict__ d_world_rot,         // [total, 4]
    const float* __restrict__ d_world_opa,         // [total, 1]
    const float* __restrict__ d_world_feat,        // [total, 3]
    // Output grads (splat-side).
    float* __restrict__ d_splat_means,
    float* __restrict__ d_splat_scales,
    float* __restrict__ d_splat_quats,
    float* __restrict__ d_splat_opa,
    float* __restrict__ d_splat_feat_dc,
    // Output grads (instance-side).
    float* __restrict__ d_inst_pos,
    float* __restrict__ d_inst_rot,
    float* __restrict__ d_inst_scl,
    float* __restrict__ d_inst_opa,
    // Scalars
    const int total, const int N,
    const float min_world_scale_log,
    const float max_world_scale_log,
    const float opac_clamp
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;

    const int inst_idx  = find_inst_idx_bwd(inst_offsets, N, g);
    const int splat_row = cand_splat_rows[g];

    // ----- Load forward inputs used in the gradient math. -----
    const float lm0 = splat_means[splat_row*3 + 0];
    const float lm1 = splat_means[splat_row*3 + 1];
    const float lm2 = splat_means[splat_row*3 + 2];
    const float ls0 = splat_scales[splat_row*3 + 0];
    const float ls1 = splat_scales[splat_row*3 + 1];
    const float ls2 = splat_scales[splat_row*3 + 2];
    const float lq0 = splat_quats[splat_row*4 + 0];
    const float lq1 = splat_quats[splat_row*4 + 1];
    const float lq2 = splat_quats[splat_row*4 + 2];
    const float lq3 = splat_quats[splat_row*4 + 3];
    const float lo_val = splat_opa[splat_row];
    const float w_ = inst_rot[inst_idx*4 + 0];
    const float x_ = inst_rot[inst_idx*4 + 1];
    const float y_ = inst_rot[inst_idx*4 + 2];
    const float z_ = inst_rot[inst_idx*4 + 3];
    const float iscl = inst_scl[inst_idx];
    const float iopa = inst_opa[inst_idx];

    // ----- Load upstream grads. -----
    const float dwm0 = d_world_means[g*3 + 0];
    const float dwm1 = d_world_means[g*3 + 1];
    const float dwm2 = d_world_means[g*3 + 2];
    const float dws0 = d_world_scales[g*3 + 0];
    const float dws1 = d_world_scales[g*3 + 1];
    const float dws2 = d_world_scales[g*3 + 2];
    const float dwr0 = d_world_rot[g*4 + 0];
    const float dwr1 = d_world_rot[g*4 + 1];
    const float dwr2 = d_world_rot[g*4 + 2];
    const float dwr3 = d_world_rot[g*4 + 3];
    const float dwo_ = d_world_opa[g];
    const float dwf0 = d_world_feat[g*3 + 0];
    const float dwf1 = d_world_feat[g*3 + 1];
    const float dwf2 = d_world_feat[g*3 + 2];

    // =====================================================================
    // (a) wm = ip + R(ir) · (lm * exp(iscl)).
    // =====================================================================
    const float es = expf(iscl);
    const float m0 = lm0 * es, m1 = lm1 * es, m2 = lm2 * es;

    // Rotation matrix R(ir):
    const float xx = x_*x_, yy = y_*y_, zz = z_*z_;
    const float xy = x_*y_, xz = x_*z_, yz_ = y_*z_;
    const float wx = w_*x_, wy = w_*y_, wz = w_*z_;
    const float R00 = 1.f - 2.f*(yy + zz);
    const float R01 = 2.f*(xy - wz);
    const float R02 = 2.f*(xz + wy);
    const float R10 = 2.f*(xy + wz);
    const float R11 = 1.f - 2.f*(xx + zz);
    const float R12 = 2.f*(yz_ - wx);
    const float R20 = 2.f*(xz - wy);
    const float R21 = 2.f*(yz_ + wx);
    const float R22 = 1.f - 2.f*(xx + yy);

    // d_m = R^T · dwm.
    const float dm0 = R00*dwm0 + R10*dwm1 + R20*dwm2;
    const float dm1 = R01*dwm0 + R11*dwm1 + R21*dwm2;
    const float dm2 = R02*dwm0 + R12*dwm1 + R22*dwm2;

    // d_lm = es * d_m.
    const float d_lm0 = es * dm0;
    const float d_lm1 = es * dm1;
    const float d_lm2 = es * dm2;

    // d_iscl from wm: <d_m, m> (since m = lm*es and ∂m/∂iscl = m).
    const float d_iscl_wm = dm0*m0 + dm1*m1 + dm2*m2;

    // d_ip = dwm.
    const float d_ip0 = dwm0, d_ip1 = dwm1, d_ip2 = dwm2;

    // d_ir from wm. Each component is sum over output dims of:
    //     dwm_i * (dR/d q_s · m)_i   for s in {w,x,y,z}.
    const float d_ir_w_wm = dwm0*(-2.f*z_*m1 + 2.f*y_*m2)
                          + dwm1*( 2.f*z_*m0 - 2.f*x_*m2)
                          + dwm2*(-2.f*y_*m0 + 2.f*x_*m1);
    const float d_ir_x_wm = dwm0*( 2.f*y_*m1 + 2.f*z_*m2)
                          + dwm1*( 2.f*y_*m0 - 4.f*x_*m1 - 2.f*w_*m2)
                          + dwm2*( 2.f*z_*m0 + 2.f*w_*m1 - 4.f*x_*m2);
    const float d_ir_y_wm = dwm0*(-4.f*y_*m0 + 2.f*x_*m1 + 2.f*w_*m2)
                          + dwm1*( 2.f*x_*m0 + 2.f*z_*m2)
                          + dwm2*(-2.f*w_*m0 + 2.f*z_*m1 - 4.f*y_*m2);
    const float d_ir_z_wm = dwm0*(-4.f*z_*m0 - 2.f*w_*m1 + 2.f*x_*m2)
                          + dwm1*( 2.f*w_*m0 - 4.f*z_*m1 + 2.f*y_*m2)
                          + dwm2*( 2.f*x_*m0 + 2.f*y_*m1);

    // =====================================================================
    // (b) ws = clamp(iscl + ls, min, max).  PyTorch clamp is inclusive on
    //     both boundaries (verified empirically).
    // =====================================================================
    const float ms0 = ((iscl+ls0 >= min_world_scale_log) && (iscl+ls0 <= max_world_scale_log)) ? 1.f : 0.f;
    const float ms1 = ((iscl+ls1 >= min_world_scale_log) && (iscl+ls1 <= max_world_scale_log)) ? 1.f : 0.f;
    const float ms2 = ((iscl+ls2 >= min_world_scale_log) && (iscl+ls2 <= max_world_scale_log)) ? 1.f : 0.f;
    const float d_ls0 = dws0 * ms0;
    const float d_ls1 = dws1 * ms1;
    const float d_ls2 = dws2 * ms2;
    const float d_iscl_ws = d_ls0 + d_ls1 + d_ls2;

    // =====================================================================
    // (c) wr = normalize_clamp1e6( F.normalize(qmul(ir, lq)) ).
    // =====================================================================
    // wr1 = qmul(ir, lq).
    const float wr1_0 = w_*lq0 - x_*lq1 - y_*lq2 - z_*lq3;
    const float wr1_1 = w_*lq1 + x_*lq0 + y_*lq3 - z_*lq2;
    const float wr1_2 = w_*lq2 - x_*lq3 + y_*lq0 + z_*lq1;
    const float wr1_3 = w_*lq3 + x_*lq2 - y_*lq1 + z_*lq0;

    const float n1_sq = wr1_0*wr1_0 + wr1_1*wr1_1 + wr1_2*wr1_2 + wr1_3*wr1_3;
    const float n1    = sqrtf(n1_sq);
    const float inv1  = 1.f / fmaxf(n1, 1e-12f);
    const float wr2_0 = wr1_0 * inv1;
    const float wr2_1 = wr1_1 * inv1;
    const float wr2_2 = wr1_2 * inv1;
    const float wr2_3 = wr1_3 * inv1;

    const float n2_sq = wr2_0*wr2_0 + wr2_1*wr2_1 + wr2_2*wr2_2 + wr2_3*wr2_3;
    const float n2    = sqrtf(n2_sq);
    const float inv2  = 1.f / fmaxf(n2, 1e-6f);
    const float wf0 = wr2_0 * inv2;
    const float wf1 = wr2_1 * inv2;
    const float wf2 = wr2_2 * inv2;
    const float wf3 = wr2_3 * inv2;

    // Backward stage 2: through wr_final = wr2 / max(n2, 1e-6).
    //   When n2 > 1e-6, df/dwr2 = (1/n2) * (I - wf wf^T).
    //   When n2 <= 1e-6, df/dwr2 = (1/eps) * I.
    float d_wr2_0, d_wr2_1, d_wr2_2, d_wr2_3;
    if (n2 > 1e-6f) {
        const float dot_dwr_wf = dwr0*wf0 + dwr1*wf1 + dwr2*wf2 + dwr3*wf3;
        const float inv_n2 = 1.f / n2;
        d_wr2_0 = inv_n2 * (dwr0 - dot_dwr_wf * wf0);
        d_wr2_1 = inv_n2 * (dwr1 - dot_dwr_wf * wf1);
        d_wr2_2 = inv_n2 * (dwr2 - dot_dwr_wf * wf2);
        d_wr2_3 = inv_n2 * (dwr3 - dot_dwr_wf * wf3);
    } else {
        d_wr2_0 = dwr0 * inv2;
        d_wr2_1 = dwr1 * inv2;
        d_wr2_2 = dwr2 * inv2;
        d_wr2_3 = dwr3 * inv2;
    }

    // Backward stage 1: through wr2 = wr1 / max(n1, 1e-12).
    float d_wr1_0, d_wr1_1, d_wr1_2, d_wr1_3;
    if (n1 > 1e-12f) {
        const float dot_dwr2_wr2 = d_wr2_0*wr2_0 + d_wr2_1*wr2_1
                                 + d_wr2_2*wr2_2 + d_wr2_3*wr2_3;
        const float inv_n1 = 1.f / n1;
        d_wr1_0 = inv_n1 * (d_wr2_0 - dot_dwr2_wr2 * wr2_0);
        d_wr1_1 = inv_n1 * (d_wr2_1 - dot_dwr2_wr2 * wr2_1);
        d_wr1_2 = inv_n1 * (d_wr2_2 - dot_dwr2_wr2 * wr2_2);
        d_wr1_3 = inv_n1 * (d_wr2_3 - dot_dwr2_wr2 * wr2_3);
    } else {
        d_wr1_0 = d_wr2_0 * inv1;
        d_wr1_1 = d_wr2_1 * inv1;
        d_wr1_2 = d_wr2_2 * inv1;
        d_wr1_3 = d_wr2_3 * inv1;
    }

    // qmul backward: d_ir, d_lq (each derived analytically from the
    // Hamilton-product expansion):
    //   wr1 = (w_lq0 - x_lq1 - y_lq2 - z_lq3,
    //          w_lq1 + x_lq0 + y_lq3 - z_lq2,
    //          w_lq2 - x_lq3 + y_lq0 + z_lq1,
    //          w_lq3 + x_lq2 - y_lq1 + z_lq0).
    const float d_ir_w_wr =  d_wr1_0*lq0 + d_wr1_1*lq1 + d_wr1_2*lq2 + d_wr1_3*lq3;
    const float d_ir_x_wr = -d_wr1_0*lq1 + d_wr1_1*lq0 - d_wr1_2*lq3 + d_wr1_3*lq2;
    const float d_ir_y_wr = -d_wr1_0*lq2 + d_wr1_1*lq3 + d_wr1_2*lq0 - d_wr1_3*lq1;
    const float d_ir_z_wr = -d_wr1_0*lq3 - d_wr1_1*lq2 + d_wr1_2*lq1 + d_wr1_3*lq0;

    const float d_lq_w   =  d_wr1_0*w_ + d_wr1_1*x_ + d_wr1_2*y_ + d_wr1_3*z_;
    const float d_lq_x   = -d_wr1_0*x_ + d_wr1_1*w_ + d_wr1_2*z_ - d_wr1_3*y_;
    const float d_lq_y   = -d_wr1_0*y_ - d_wr1_1*z_ + d_wr1_2*w_ + d_wr1_3*x_;
    const float d_lq_z   = -d_wr1_0*z_ + d_wr1_1*y_ - d_wr1_2*x_ + d_wr1_3*w_;

    // =====================================================================
    // (d) wo = clamp(lo + io, -c, c).
    // =====================================================================
    const float opa_sum = iopa + lo_val;
    const float mo = ((opa_sum >= -opac_clamp) && (opa_sum <= opac_clamp)) ? 1.f : 0.f;
    const float d_lo_total = dwo_ * mo;
    const float d_io_total = dwo_ * mo;

    // =====================================================================
    // (e) wf = lf  =>  d_lf = dwf.
    // =====================================================================
    const float d_lf0 = dwf0, d_lf1 = dwf1, d_lf2 = dwf2;

    // =====================================================================
    // Combine instance-side iscl + ir contributions from multiple outputs.
    // =====================================================================
    const float d_iscl_total = d_iscl_wm + d_iscl_ws;
    const float d_ir_w_total = d_ir_w_wm + d_ir_w_wr;
    const float d_ir_x_total = d_ir_x_wm + d_ir_x_wr;
    const float d_ir_y_total = d_ir_y_wm + d_ir_y_wr;
    const float d_ir_z_total = d_ir_z_wm + d_ir_z_wr;

    // =====================================================================
    // Scatter atomicAdd to splat-side accumulators.
    // =====================================================================
    atomicAdd(&d_splat_means[splat_row*3 + 0], d_lm0);
    atomicAdd(&d_splat_means[splat_row*3 + 1], d_lm1);
    atomicAdd(&d_splat_means[splat_row*3 + 2], d_lm2);
    atomicAdd(&d_splat_scales[splat_row*3 + 0], d_ls0);
    atomicAdd(&d_splat_scales[splat_row*3 + 1], d_ls1);
    atomicAdd(&d_splat_scales[splat_row*3 + 2], d_ls2);
    atomicAdd(&d_splat_quats[splat_row*4 + 0], d_lq_w);
    atomicAdd(&d_splat_quats[splat_row*4 + 1], d_lq_x);
    atomicAdd(&d_splat_quats[splat_row*4 + 2], d_lq_y);
    atomicAdd(&d_splat_quats[splat_row*4 + 3], d_lq_z);
    atomicAdd(&d_splat_opa[splat_row],         d_lo_total);
    atomicAdd(&d_splat_feat_dc[splat_row*3 + 0], d_lf0);
    atomicAdd(&d_splat_feat_dc[splat_row*3 + 1], d_lf1);
    atomicAdd(&d_splat_feat_dc[splat_row*3 + 2], d_lf2);

    // =====================================================================
    // Scatter atomicAdd to instance-side accumulators.
    // =====================================================================
    atomicAdd(&d_inst_pos[inst_idx*3 + 0], d_ip0);
    atomicAdd(&d_inst_pos[inst_idx*3 + 1], d_ip1);
    atomicAdd(&d_inst_pos[inst_idx*3 + 2], d_ip2);
    atomicAdd(&d_inst_rot[inst_idx*4 + 0], d_ir_w_total);
    atomicAdd(&d_inst_rot[inst_idx*4 + 1], d_ir_x_total);
    atomicAdd(&d_inst_rot[inst_idx*4 + 2], d_ir_y_total);
    atomicAdd(&d_inst_rot[inst_idx*4 + 3], d_ir_z_total);
    atomicAdd(&d_inst_scl[inst_idx], d_iscl_total);
    atomicAdd(&d_inst_opa[inst_idx], d_io_total);
}

// ----------------------------------------------------------------------------
// Host-side launcher.
//
// Returns 9 tensors in the order:
//   d_splat_means, d_splat_scales, d_splat_quats, d_splat_opa, d_splat_feat_dc,
//   d_inst_pos,    d_inst_rot,    d_inst_scl,    d_inst_opa
//
// ----------------------------------------------------------------------------
std::vector<torch::Tensor> assemble_fused_bwd(
    torch::Tensor splat_means,
    torch::Tensor splat_scales,
    torch::Tensor splat_quats,
    torch::Tensor splat_opa,
    torch::Tensor splat_feat_dc,
    torch::Tensor inst_pos,
    torch::Tensor inst_rot,
    torch::Tensor inst_scl,
    torch::Tensor inst_opa,
    torch::Tensor cand_splat_rows,
    torch::Tensor inst_offsets,
    torch::Tensor d_world_means,
    torch::Tensor d_world_scales,
    torch::Tensor d_world_rot,
    torch::Tensor d_world_opa,
    torch::Tensor d_world_feat,
    double min_world_scale_log,
    double max_world_scale_log,
    double opac_clamp
) {
    TORCH_CHECK(splat_means.is_cuda(),         "splat_means must be CUDA");
    TORCH_CHECK(d_world_means.is_cuda(),       "d_world_means must be CUDA");
    TORCH_CHECK(cand_splat_rows.dtype() == torch::kInt32, "cand_splat_rows must be int32");
    TORCH_CHECK(inst_offsets.dtype()    == torch::kInt32, "inst_offsets must be int32");

    // PyTorch sometimes hands us non-contiguous upstream gradients —
    // e.g. ``loss += k * out.sum()`` produces a stride-0 expanded
    // ``ones_like`` for that output's gradient. The kernel indexes via
    // raw ``data_ptr<float>()`` and assumes contiguous row-major layout,
    // so non-contiguous strides read garbage memory. Force-materialise.
    d_world_means  = d_world_means.contiguous();
    d_world_scales = d_world_scales.contiguous();
    d_world_rot    = d_world_rot.contiguous();
    d_world_opa    = d_world_opa.contiguous();
    d_world_feat   = d_world_feat.contiguous();
    // Forward inputs may also arrive non-contiguous after autograd
    // detach/clone chains (rare, but cheap to defend).
    splat_means  = splat_means.contiguous();
    splat_scales = splat_scales.contiguous();
    splat_quats  = splat_quats.contiguous();
    splat_opa    = splat_opa.contiguous();
    inst_rot     = inst_rot.contiguous();
    inst_scl     = inst_scl.contiguous();
    inst_opa     = inst_opa.contiguous();
    cand_splat_rows = cand_splat_rows.contiguous();
    inst_offsets    = inst_offsets.contiguous();

    const int N     = static_cast<int>(inst_offsets.size(0)) - 1;
    const int total = static_cast<int>(cand_splat_rows.size(0));

    auto d_splat_means    = torch::zeros_like(splat_means);
    auto d_splat_scales   = torch::zeros_like(splat_scales);
    auto d_splat_quats    = torch::zeros_like(splat_quats);
    auto d_splat_opa      = torch::zeros_like(splat_opa);
    auto d_splat_feat_dc  = torch::zeros_like(splat_feat_dc);
    auto d_inst_pos       = torch::zeros_like(inst_pos);
    auto d_inst_rot       = torch::zeros_like(inst_rot);
    auto d_inst_scl       = torch::zeros_like(inst_scl);
    auto d_inst_opa       = torch::zeros_like(inst_opa);

    if (total == 0) {
        return {
            d_splat_means, d_splat_scales, d_splat_quats, d_splat_opa, d_splat_feat_dc,
            d_inst_pos, d_inst_rot, d_inst_scl, d_inst_opa,
        };
    }

    const int block = BLOCK_SIZE_ASSEMBLE;
    const int grid  = ceil_div(total, block);

    auto stream = at::cuda::getCurrentCUDAStream();
    assemble_fused_bwd_kernel<<<grid, block, 0, stream>>>(
        splat_means.data_ptr<float>(),
        splat_scales.data_ptr<float>(),
        splat_quats.data_ptr<float>(),
        splat_opa.data_ptr<float>(),
        inst_rot.data_ptr<float>(),
        inst_scl.data_ptr<float>(),
        inst_opa.data_ptr<float>(),
        cand_splat_rows.data_ptr<int32_t>(),
        inst_offsets.data_ptr<int32_t>(),
        d_world_means.data_ptr<float>(),
        d_world_scales.data_ptr<float>(),
        d_world_rot.data_ptr<float>(),
        d_world_opa.data_ptr<float>(),
        d_world_feat.data_ptr<float>(),
        d_splat_means.data_ptr<float>(),
        d_splat_scales.data_ptr<float>(),
        d_splat_quats.data_ptr<float>(),
        d_splat_opa.data_ptr<float>(),
        d_splat_feat_dc.data_ptr<float>(),
        d_inst_pos.data_ptr<float>(),
        d_inst_rot.data_ptr<float>(),
        d_inst_scl.data_ptr<float>(),
        d_inst_opa.data_ptr<float>(),
        total, N,
        static_cast<float>(min_world_scale_log),
        static_cast<float>(max_world_scale_log),
        static_cast<float>(opac_clamp)
    );

    return {
        d_splat_means, d_splat_scales, d_splat_quats, d_splat_opa, d_splat_feat_dc,
        d_inst_pos, d_inst_rot, d_inst_scl, d_inst_opa,
    };
}

}  // namespace scion_cuda
