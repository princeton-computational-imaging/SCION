// pybind11 module entry point for scion_cuda._C.
//
// `assemble_fused` forward + backward: the codebook assembly hot path.
// Other kernels were measured and dropped: their Python paths cost < 2% of
// a 30k-step training run (garden, RTX 5090).

#include <torch/extension.h>
#include <vector>

namespace scion_cuda {

// Forward declarations of the kernel launchers (defined in src/csrc/*.cu).
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
    double opac_clamp);

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
    double opac_clamp);

}  // namespace scion_cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assemble_fused", &scion_cuda::assemble_fused,
          "Fused gather/transform for codebook assembly (forward).");
    m.def("assemble_fused_bwd", &scion_cuda::assemble_fused_bwd,
          "Gradient kernel for assemble_fused.");
}
