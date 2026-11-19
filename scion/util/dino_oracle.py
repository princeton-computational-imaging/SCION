"""DINO Feature Oracle for codebook initialization.

Holds pre-computed DINOv2/v3 features at a set of world-space anchor
positions and provides query-time lookup from arbitrary world-space
positions to DINO features, with either brute-force kNN or a sparse
voxel backend. Used only at codebook init; no training-time feature
recomputation lives on this class.

Usage:
    oracle = DinoFeatureOracle.from_training_views(positions, scene_dir)
    features = oracle.lookup(query_positions)       # [M, D]
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DinoFeatureOracle(nn.Module):
    """Frozen DINO feature oracle backed by world-space lookup.

    Stores anchor positions ``[N_orig, 3]`` and their pre-computed DINO
    features ``[N_orig, D]``. Lookup supports:

    - ``brute_force``: chunked ``torch.cdist`` over all original points.
    - ``voxel_hash``: sparse voxel index with vectorized neighborhood search,
      plus brute-force fallback for out-of-neighborhood queries.
    """

    BACKEND_BRUTE_FORCE = 0
    BACKEND_VOXEL_HASH = 1

    def __init__(
        self,
        positions: Tensor,
        features: Tensor,
        k: int = 1,
        device: str = "cuda",
        backend: str = "brute_force",
        voxel_size: float = 0.0,
        voxel_neighbor_radius: int = 1,
    ) -> None:
        super().__init__()
        assert positions.shape[0] == features.shape[0], (
            f"Position/feature count mismatch: {positions.shape[0]} vs {features.shape[0]}"
        )
        if backend not in {"brute_force", "voxel_hash"}:
            raise ValueError(f"Unknown DINO oracle backend: {backend}")
        if voxel_neighbor_radius < 0:
            raise ValueError("voxel_neighbor_radius must be >= 0")

        self._k = int(k)
        self._backend_name = backend
        self._backend_id = (
            self.BACKEND_VOXEL_HASH if backend == "voxel_hash" else self.BACKEND_BRUTE_FORCE
        )

        pos = positions.float().contiguous().cpu()
        feat = features.float().contiguous().cpu()

        self.register_buffer("_positions", pos)
        self.register_buffer("_features", feat)

        self._voxel_size = float(
            voxel_size if voxel_size > 0 else self._auto_voxel_size(self._positions)
        )
        self._voxel_neighbor_radius = int(voxel_neighbor_radius)
        # Per-chunk candidate-pair budget.  Each pair allocates several
        # long/float intermediates (~64 B), so 4M pairs ≈ 256 MB peak —
        # safe even on a fairly full GPU.
        self._voxel_candidate_budget = 4_000_000
        # Resolved at index build time once we know the densest cell.
        self._voxel_chunk_size: int = 2048

        # Voxel backend buffers are kept empty for brute-force mode.
        self.register_buffer("_grid_min", torch.zeros(3, dtype=torch.long))
        self.register_buffer("_grid_shape", torch.ones(3, dtype=torch.long))
        self.register_buffer("_neighbor_offsets", torch.zeros(1, 3, dtype=torch.long))
        self.register_buffer("_unique_cell_ids", torch.zeros(0, dtype=torch.long))
        self.register_buffer("_cell_starts", torch.zeros(0, dtype=torch.long))
        self.register_buffer("_cell_counts", torch.zeros(0, dtype=torch.long))
        self.register_buffer("_sorted_positions", torch.empty(0, 3), persistent=False)
        self.register_buffer("_sorted_features", torch.empty(0, feat.shape[-1]), persistent=False)

        if self._backend_id == self.BACKEND_VOXEL_HASH:
            self._build_voxel_index()
            self._compute_safe_chunk_size()

        self.to(device)

    @property
    def feature_dim(self) -> int:
        """Dimensionality D of DINO features."""
        return self._features.shape[-1]

    @property
    def num_points(self) -> int:
        return self._features.shape[0]

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def voxel_size(self) -> float:
        return self._voxel_size

    def to(self, *args, **kwargs) -> "DinoFeatureOracle":
        super().to(*args, **kwargs)
        return self

    @staticmethod
    def _auto_voxel_size(positions: Tensor) -> float:
        """Estimate a default voxel size from point density."""
        bbox_min = positions.amin(dim=0)
        bbox_max = positions.amax(dim=0)
        bbox_extent = (bbox_max - bbox_min).clamp(min=1e-6)
        volume = float((bbox_extent[0] * bbox_extent[1] * bbox_extent[2]).item())
        density_spacing = (volume / max(int(positions.shape[0]), 1)) ** (1.0 / 3.0)
        return max(density_spacing, 1e-4)

    @staticmethod
    def _build_neighbor_offsets(radius: int) -> Tensor:
        coords = torch.arange(-radius, radius + 1, dtype=torch.long)
        return torch.cartesian_prod(coords, coords, coords).contiguous()

    def _build_voxel_index(self) -> None:
        """Build a sparse voxel index on CPU, then move with the module."""
        voxel_coords = torch.floor(self._positions / self._voxel_size).to(torch.long)
        grid_min = voxel_coords.amin(dim=0)
        grid_max = voxel_coords.amax(dim=0)
        grid_shape = (grid_max - grid_min + 1).to(torch.long)
        linear_ids = self._linearize_coords_static(voxel_coords, grid_min, grid_shape)

        perm = torch.argsort(linear_ids)
        sorted_ids = linear_ids[perm]

        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        starts = counts.cumsum(0) - counts

        self._grid_min = grid_min
        self._grid_shape = grid_shape
        self._neighbor_offsets = self._build_neighbor_offsets(self._voxel_neighbor_radius)
        self._unique_cell_ids = unique_ids
        self._cell_starts = starts
        self._cell_counts = counts
        self._positions = self._positions[perm]
        self._features = self._features[perm]

    def _compute_safe_chunk_size(self) -> None:
        """Pick a per-chunk query count that respects the candidate budget.

        Reads ``_cell_counts.max()`` once at index build time (a one-off
        host sync that's fine outside the training hot path).  We then
        derive the worst-case candidate count per query from the densest
        cell × number of neighbour cells, and shrink the chunk size so
        the per-chunk allocation never exceeds ``_voxel_candidate_budget``.

        For typical scenes the densest cell holds only a handful of
        points, so this returns the 2048 cap.  For scenes with hot spots
        (the "killed during MH" case) we automatically fall back to a
        much smaller chunk so the lookup runs without OOMing.
        """
        if self._cell_counts.numel() == 0:
            self._voxel_chunk_size = 2048
            return
        max_per_cell = int(self._cell_counts.max().item())
        neighbor_count = int(self._neighbor_offsets.shape[0])
        worst_case_per_query = max(max_per_cell * neighbor_count, 1)
        safe = self._voxel_candidate_budget // worst_case_per_query
        self._voxel_chunk_size = int(max(1, min(2048, safe)))

    @staticmethod
    def _linearize_coords_static(
        coords: Tensor,
        grid_min: Tensor,
        grid_shape: Tensor,
    ) -> Tensor:
        shifted = coords.to(torch.long) - grid_min.unsqueeze(0)
        sx = shifted[:, 0]
        sy = shifted[:, 1]
        sz = shifted[:, 2]
        stride_y = grid_shape[1]
        stride_x = grid_shape[0]
        return sx + stride_x * (sy + stride_y * sz)

    @torch.jit.export
    def _linearize_query_coords(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        shifted = coords.to(torch.long) - self._grid_min.unsqueeze(0)
        valid = ((shifted >= 0) & (shifted < self._grid_shape.unsqueeze(0))).all(dim=-1)
        shifted = shifted.clamp(min=0)
        sx = shifted[:, 0]
        sy = shifted[:, 1]
        sz = shifted[:, 2]
        stride_y = self._grid_shape[1]
        stride_x = self._grid_shape[0]
        ids = sx + stride_x * (sy + stride_y * sz)
        return ids, valid

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def _auto_chunk_size(self, budget_gb: float = 1.0) -> int:
        """Compute chunk_size so brute-force cdist allocates at most budget."""
        max_elements = int(budget_gb * (1024 ** 3) / 4)
        return max(64, max_elements // max(int(self._positions.shape[0]), 1))

    def _auto_lookup_chunk_size(self) -> int:
        if self._backend_id == 1:
            # Statically derived from the densest voxel cell at index
            # build time (see ``_compute_safe_chunk_size``) so a single
            # chunk never blows past ``_voxel_candidate_budget`` (no
            # data-dependent ``.item()`` subchunking at lookup time).
            return self._voxel_chunk_size
        return self._auto_chunk_size()

    @torch.jit.export
    def forward(self, query_positions: Tensor, chunk_size: int = 0) -> Tensor:
        q = query_positions.detach().to(self._positions.device).float()
        if q.numel() == 0:
            return torch.empty(
                0, self.feature_dim, device=self._positions.device, dtype=self._features.dtype
            )

        if chunk_size <= 0:
            chunk_size = self._auto_lookup_chunk_size()

        parts = []
        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            if self._backend_id == 1 and self._k == 1:
                parts.append(self._lookup_voxel_hash_chunk(q[start:end]))
            else:
                parts.append(self._lookup_brute_force_chunk(q[start:end]))
        return torch.cat(parts, dim=0)

    @torch.jit.export
    def lookup(self, query_positions: Tensor, chunk_size: int = 0) -> Tensor:
        return self.forward(query_positions, chunk_size)

    @torch.jit.export
    def _lookup_brute_force_chunk(self, q: Tensor) -> Tensor:
        if self._k == 1:
            dists = torch.cdist(q, self._positions)
            return self._features[dists.argmin(dim=-1)]

        dists = torch.cdist(q, self._positions)
        topk_dists, topk_idx = dists.topk(self._k, largest=False)
        weights = 1.0 / topk_dists.clamp(min=1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        neighbour_feats = self._features[topk_idx]
        return (neighbour_feats * weights.unsqueeze(-1)).sum(dim=1)

    @torch.jit.export
    def _lookup_voxel_hash_chunk(self, q: Tensor) -> Tensor:
        """Sparse voxel neighborhood lookup for k=1, with brute-force fallback.

        Sync-free hot path: every operation runs on-device with no
        ``.item()`` / ``.any()`` reads.  The single unavoidable
        device->host sync is the brute-force fallback for queries with
        no neighbour in the search radius (rare for typical scenes).
        """
        num_queries = q.shape[0]
        feat_dim = self.feature_dim
        if num_queries == 0:
            return torch.zeros(0, feat_dim, device=q.device, dtype=self._features.dtype)

        out = torch.zeros(num_queries, feat_dim, device=q.device, dtype=self._features.dtype)

        query_cells = torch.floor(q / self._voxel_size).to(torch.long)
        neighbor_cells = query_cells.unsqueeze(1) + self._neighbor_offsets.unsqueeze(0)
        flat_cells = neighbor_cells.reshape(-1, 3)

        flat_ids, flat_valid = self._linearize_query_coords(flat_cells)
        num_unique = int(self._unique_cell_ids.shape[0])
        search_idx = torch.searchsorted(self._unique_cell_ids, flat_ids)
        safe_idx = search_idx.clamp(max=max(num_unique - 1, 0))

        matches = flat_valid & (search_idx < num_unique)
        if num_unique > 0:
            matches = matches & (self._unique_cell_ids[safe_idx] == flat_ids)

        starts = torch.zeros_like(safe_idx)
        counts = torch.zeros_like(safe_idx)
        starts[matches] = self._cell_starts[safe_idx[matches]]
        counts[matches] = self._cell_counts[safe_idx[matches]]

        num_neighbor_cells = int(self._neighbor_offsets.shape[0])
        num_cells_flat = int(counts.shape[0])

        # Expand (cell, count) -> (point_idx, owner) without any host sync.
        # `repeat_interleave(empty)` is well-defined and yields an empty tensor,
        # so the entire path stays valid even when no query has a hit.
        owner = torch.repeat_interleave(
            torch.arange(num_cells_flat, device=q.device, dtype=torch.long), counts
        )
        cell_prefix = counts.cumsum(0) - counts
        # Local offset within each cell's slice of `_sorted_positions`.
        arange_total = torch.arange(owner.shape[0], device=q.device, dtype=torch.long)
        local_offset = arange_total - torch.repeat_interleave(cell_prefix, counts)
        point_idx = torch.repeat_interleave(starts, counts) + local_offset

        query_idx = torch.div(owner, num_neighbor_cells, rounding_mode="floor")
        candidate_pos = self._positions[point_idx]
        dists = (candidate_pos - q[query_idx]).pow(2).sum(dim=-1)

        best_dist = torch.full((num_queries,), math.inf, device=q.device)
        best_dist.scatter_reduce_(0, query_idx, dists, reduce="amin", include_self=True)

        sentinel = int(self._positions.shape[0])
        is_best = dists <= best_dist[query_idx] + 1e-12
        candidate_ids = torch.where(
            is_best,
            point_idx,
            torch.full_like(point_idx, sentinel),
        )

        best_point = torch.full(
            (num_queries,), sentinel, device=q.device, dtype=torch.long
        )
        best_point.scatter_reduce_(0, query_idx, candidate_ids, reduce="amin", include_self=True)

        has_local_match = best_point < sentinel
        out[has_local_match] = self._features[best_point[has_local_match]]

        # Brute-force fallback for queries that fell outside any indexed
        # voxel neighbourhood.  This is the only remaining sync per chunk
        # and is rare in practice (most queries land inside the index).
        missing_mask = ~has_local_match
        missing_q = q[missing_mask]
        if missing_q.shape[0] > 0:
            out[missing_mask] = self._lookup_brute_force_chunk(missing_q)

        return out

    # ------------------------------------------------------------------
    # Construction from training views
    # ------------------------------------------------------------------

    @staticmethod
    def from_training_views(
        positions: Tensor,
        scene_dir: str,
        *,
        dino_model: str = "dinov3_vitl16",
        resolution: int = 832,
        pca_dim: int = 64,
        distance_alpha: float = 0.0,
        k: int = 1,
        device: str = "cuda",
        backend: str = "brute_force",
        voxel_size: float = 0.0,
        voxel_neighbor_radius: int = 1,
        scene=None,
    ) -> DinoFeatureOracle:
        """Build an oracle by projecting ``positions`` directly into training views.

        Each query position is projected into every training image, the
        per-image DINO feature map is sampled bilinearly, and the
        contributions are aggregated across views (optionally with an
        ``exp(-α·dist)`` per-view distance falloff).  An optional PCA reduces
        the feature dim to ``pca_dim`` before storage.

        ``scene_dir`` should contain ``sparse/0/`` (COLMAP) or
        ``transforms_train.json`` (Blender). Only the train split of
        ``scene`` (a :class:`~scion.data.Scene` in the training
        frame) is used; without one, ``scene_dir`` is loaded with the
        default split (:func:`scion.data.load_scene`).

        Args:
            positions: ``[N, 3]`` world-space positions to feature-tag.
            scene_dir: Scene directory (``data/{scene}``).
            dino_model: DINO backbone name (registry or timm).
            resolution: DINO input resolution (largest image dim).
            pca_dim: PCA components for feature reduction (0 = skip).
            distance_alpha: Per-view distance falloff weight (0 = uniform).
                The per-image feature cache is ``<scene_dir>/dino`` (also
                written by ``scripts/preprocess_dino.py``).
            k, device, backend, voxel_size, voxel_neighbor_radius:
                Forwarded to the :class:`DinoFeatureOracle` constructor.
        """
        from pathlib import Path

        from scion.util.dino_init import (
            pca_reduce,
            project_features_at_positions,
            train_feature_maps,
            train_views,
        )

        scene_path = Path(scene_dir)
        proj_device = device if torch.cuda.is_available() else "cpu"
        if scene is None:
            from scion.data import load_scene

            scene = load_scene(scene_path)
        viewmats, Ks, _paths, W, H = train_views(scene, device=proj_device)
        print(
            f"[DinoFeatureOracle.from_training_views] {len(viewmats)} train cameras, "
            f"{W}x{H}, scene={scene_dir}"
        )
        features_2d = train_feature_maps(
            scene, scene_path, model_name=dino_model, resolution=resolution,
            pca_dim=pca_dim, device=proj_device,
        )

        feats = project_features_at_positions(
            positions,
            viewmats,
            Ks,
            features_2d,
            W,
            H,
            distance_alpha=distance_alpha,
            device=proj_device,
        )

        if pca_dim > 0 and pca_dim < feats.shape[-1]:
            feats = pca_reduce(feats, pca_dim)

        print(
            f"[DinoFeatureOracle.from_training_views] {positions.shape[0]:,} positions, "
            f"feature dim {feats.shape[-1]}, distance_alpha={distance_alpha}"
        )
        return DinoFeatureOracle(
            positions.detach().cpu(),
            feats.detach().cpu(),
            k=k,
            device=device,
            backend=backend,
            voxel_size=voxel_size,
            voxel_neighbor_radius=voxel_neighbor_radius,
        )
