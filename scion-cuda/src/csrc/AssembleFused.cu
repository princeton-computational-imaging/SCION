// Forward kernel for assemble_fused. See docs/cuda_architecture.md.
//
// One thread per world-Gaussian (total = inst_offsets[N]). Reads:
//   - splat-side buffers via splat_row = cand_splat_rows[g]
//   - instance-side buffers via inst_idx (decoded via binary search of
//     inst_offsets; N+1 is small relative to total so log2(N) is cheap)
// Writes the five world-space buffers.
//
// Quaternion convention: wxyz Hamilton product, matching
// scion.util.transformation.quat_mul + quat_rotate. The rotation
// formula used in quat_rotate is the cross-product form:
//     v' = v + 2*w*(qvec × v) + 2*(qvec × (qvec × v))
// which we expand to the explicit 3x3 rotation matrix below.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "Common.cuh"

namespace scion_cuda {

// Apply R(q) · v where q = (w, x, y, z) wxyz. Matches quat_rotate exactly.
__device__ __forceinline__ void quat_rotate_v3(
    const float q[4],
    const float v[3],
    float out[3]
) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    // R = I + 2*w*[qvec]_x + 2*[qvec]_x^2, expanded:
    const float xx = x*x, yy = y*y, zz = z*z;
    const float xy = x*y, xz = x*z, yz = y*z;
    const float wx = w*x, wy = w*y, wz = w*z;

    const float R00 = 1.f - 2.f*(yy + zz);
    const float R01 = 2.f*(xy - wz);
    const float R02 = 2.f*(xz + wy);
    const float R10 = 2.f*(xy + wz);
    const float R11 = 1.f - 2.f*(xx + zz);
    const float R12 = 2.f*(yz - wx);
    const float R20 = 2.f*(xz - wy);
    const float R21 = 2.f*(yz + wx);
    const float R22 = 1.f - 2.f*(xx + yy);

    out[0] = R00*v[0] + R01*v[1] + R02*v[2];
    out[1] = R10*v[0] + R11*v[1] + R12*v[2];
    out[2] = R20*v[0] + R21*v[1] + R22*v[2];
}

// Hamilton product q1 ⊗ q2 (wxyz convention). Matches quat_mul.
__device__ __forceinline__ void quat_mul(
    const float q1[4],
    const float q2[4],
    float out[4]
) {
    const float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    const float w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
    out[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    out[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    out[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    out[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

// Binary-search inst_offsets[] for the inst_idx containing g.
// Returns the smallest i such that inst_offsets[i+1] > g.
__device__ __forceinline__ int find_inst_idx(
    const int32_t* __restrict__ inst_offsets,
    int N,
    int g
) {
    int lo = 0, hi = N;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (inst_offsets[mid + 1] <= g) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void assemble_fused_kernel(
    // Splat buffers (read-only)
    const float* __restrict__ splat_means,        // [T_max, 3]
    const float* __restrict__ splat_scales,       // [T_max, 3]
    const float* __restrict__ splat_quats,        // [T_max, 4]
    const float* __restrict__ splat_opa,          // [T_max, 1]
    const float* __restrict__ splat_feat_dc,      // [T_max, 3]
    // Instance buffers (read-only)
    const float* __restrict__ inst_pos,           // [N, 3]
    const float* __restrict__ inst_rot,           // [N, 4]
    const float* __restrict__ inst_scl,           // [N, 1]
    const float* __restrict__ inst_opa,           // [N, 1]
    // Layout
    const int32_t* __restrict__ cand_splat_rows,  // [total]
    const int32_t* __restrict__ inst_offsets,     // [N+1]
    // Outputs (per-Gaussian)
    float* __restrict__ world_means,              // [total, 3]
    float* __restrict__ world_scales,             // [total, 3]
    float* __restrict__ world_rot,                // [total, 4]
    float* __restrict__ world_opa,                // [total, 1]
    float* __restrict__ world_feat,               // [total, 3]
    // Scalars
    const int total, const int N,
    const float min_world_scale_log,
    const float max_world_scale_log,
    const float opac_clamp
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;

    // 1. Decode (inst_idx, splat_row).
    const int inst_idx = find_inst_idx(inst_offsets, N, g);
    const int splat_row = cand_splat_rows[g];

    // 2. Load splat fields.
    float lm[3] = {
        splat_means[splat_row*3 + 0],
        splat_means[splat_row*3 + 1],
        splat_means[splat_row*3 + 2],
    };
    float ls[3] = {
        splat_scales[splat_row*3 + 0],
        splat_scales[splat_row*3 + 1],
        splat_scales[splat_row*3 + 2],
    };
    float lq[4] = {
        splat_quats[splat_row*4 + 0],
        splat_quats[splat_row*4 + 1],
        splat_quats[splat_row*4 + 2],
        splat_quats[splat_row*4 + 3],
    };
    const float lo_ = splat_opa[splat_row];
    float lf[3] = {
        splat_feat_dc[splat_row*3 + 0],
        splat_feat_dc[splat_row*3 + 1],
        splat_feat_dc[splat_row*3 + 2],
    };

    // 3. Load instance fields.
    float ip[3] = {
        inst_pos[inst_idx*3 + 0],
        inst_pos[inst_idx*3 + 1],
        inst_pos[inst_idx*3 + 2],
    };
    float ir[4] = {
        inst_rot[inst_idx*4 + 0],
        inst_rot[inst_idx*4 + 1],
        inst_rot[inst_idx*4 + 2],
        inst_rot[inst_idx*4 + 3],
    };
    const float iscl = inst_scl[inst_idx];
    const float iopa = inst_opa[inst_idx];

    // 4. world_means = ip + rotate(ir, lm * exp(iscl))
    const float es = expf(iscl);
    const float local_scaled[3] = {lm[0]*es, lm[1]*es, lm[2]*es};
    float rotated[3];
    quat_rotate_v3(ir, local_scaled, rotated);
    world_means[g*3 + 0] = ip[0] + rotated[0];
    world_means[g*3 + 1] = ip[1] + rotated[1];
    world_means[g*3 + 2] = ip[2] + rotated[2];

    // 5. world_scales = clamp(iscl + ls, min, max)
    world_scales[g*3 + 0] = fminf(fmaxf(iscl + ls[0], min_world_scale_log), max_world_scale_log);
    world_scales[g*3 + 1] = fminf(fmaxf(iscl + ls[1], min_world_scale_log), max_world_scale_log);
    world_scales[g*3 + 2] = fminf(fmaxf(iscl + ls[2], min_world_scale_log), max_world_scale_log);

    // 6. world_rot = normalize(quat_mul(ir, lq)). Reference does:
    //    a) F.normalize(qmul) — divides by max(||qmul||, 1e-12)
    //    b) further divides by clamp(||result||, min=1e-6)
    //    For non-degenerate inputs ||qmul|| ~ 1 product of input unit
    //    quats, and (b) is a no-op. We mirror the double-normalize for
    //    bit-exact parity.
    float wq[4];
    quat_mul(ir, lq, wq);
    {
        const float n2 = wq[0]*wq[0] + wq[1]*wq[1] + wq[2]*wq[2] + wq[3]*wq[3];
        const float n  = sqrtf(n2);
        const float inv1 = 1.f / fmaxf(n, 1e-12f);
        wq[0] *= inv1; wq[1] *= inv1; wq[2] *= inv1; wq[3] *= inv1;
        const float n2b = wq[0]*wq[0] + wq[1]*wq[1] + wq[2]*wq[2] + wq[3]*wq[3];
        const float nb  = sqrtf(n2b);
        const float inv2 = 1.f / fmaxf(nb, 1e-6f);
        wq[0] *= inv2; wq[1] *= inv2; wq[2] *= inv2; wq[3] *= inv2;
    }
    world_rot[g*4 + 0] = wq[0];
    world_rot[g*4 + 1] = wq[1];
    world_rot[g*4 + 2] = wq[2];
    world_rot[g*4 + 3] = wq[3];

    // 7. world_opa = clamp(iopa + lo_, -opac_clamp, opac_clamp)
    world_opa[g] = fminf(fmaxf(iopa + lo_, -opac_clamp), opac_clamp);

    // 8. world_feat = lf (instances share the candidate's colour)
    world_feat[g*3 + 0] = lf[0];
    world_feat[g*3 + 1] = lf[1];
    world_feat[g*3 + 2] = lf[2];
}

// Host-side launcher.
std::vector<torch::Tensor> assemble_fused(
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
    double min_world_scale_log,
    double max_world_scale_log,
    double opac_clamp
) {
    TORCH_CHECK(splat_means.is_cuda(), "splat_means must be CUDA");
    TORCH_CHECK(splat_means.dtype()  == torch::kFloat32, "splat_means must be fp32");
    TORCH_CHECK(splat_scales.dtype() == torch::kFloat32, "splat_scales must be fp32");
    TORCH_CHECK(splat_quats.dtype()  == torch::kFloat32, "splat_quats must be fp32");
    TORCH_CHECK(splat_opa.dtype()    == torch::kFloat32, "splat_opa must be fp32");
    TORCH_CHECK(splat_feat_dc.dtype()== torch::kFloat32, "splat_feat_dc must be fp32");
    TORCH_CHECK(inst_pos.dtype()     == torch::kFloat32, "inst_pos must be fp32");
    TORCH_CHECK(inst_rot.dtype()     == torch::kFloat32, "inst_rot must be fp32");
    TORCH_CHECK(inst_scl.dtype()     == torch::kFloat32, "inst_scl must be fp32");
    TORCH_CHECK(inst_opa.dtype()     == torch::kFloat32, "inst_opa must be fp32");
    TORCH_CHECK(cand_splat_rows.dtype() == torch::kInt32, "cand_splat_rows must be int32");
    TORCH_CHECK(inst_offsets.dtype()    == torch::kInt32, "inst_offsets must be int32");
    TORCH_CHECK(cand_splat_rows.is_contiguous(), "cand_splat_rows must be contiguous");
    TORCH_CHECK(inst_offsets.is_contiguous(),    "inst_offsets must be contiguous");

    const int N     = static_cast<int>(inst_offsets.size(0)) - 1;
    const int total = static_cast<int>(cand_splat_rows.size(0));
    auto opts = splat_means.options();
    auto world_means  = torch::empty({total, 3}, opts);
    auto world_scales = torch::empty({total, 3}, opts);
    auto world_rot    = torch::empty({total, 4}, opts);
    auto world_opa    = torch::empty({total, 1}, opts);
    auto world_feat   = torch::empty({total, 3}, opts);

    if (total == 0) {
        return {world_means, world_scales, world_rot, world_opa, world_feat};
    }

    const int block = BLOCK_SIZE_ASSEMBLE;
    const int grid  = ceil_div(total, block);

    auto stream = at::cuda::getCurrentCUDAStream();
    assemble_fused_kernel<<<grid, block, 0, stream>>>(
        splat_means.data_ptr<float>(),  splat_scales.data_ptr<float>(),
        splat_quats.data_ptr<float>(),  splat_opa.data_ptr<float>(),
        splat_feat_dc.data_ptr<float>(),
        inst_pos.data_ptr<float>(),     inst_rot.data_ptr<float>(),
        inst_scl.data_ptr<float>(),     inst_opa.data_ptr<float>(),
        cand_splat_rows.data_ptr<int32_t>(),
        inst_offsets.data_ptr<int32_t>(),
        world_means.data_ptr<float>(),  world_scales.data_ptr<float>(),
        world_rot.data_ptr<float>(),    world_opa.data_ptr<float>(),
        world_feat.data_ptr<float>(),
        total, N,
        static_cast<float>(min_world_scale_log),
        static_cast<float>(max_world_scale_log),
        static_cast<float>(opac_clamp)
    );

    return {world_means, world_scales, world_rot, world_opa, world_feat};
}

}  // namespace scion_cuda
