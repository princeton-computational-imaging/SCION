# Compression levels

SCION stores a trained scene at one of three levels. Each level does all
the work of the one before it. A run evaluates **every level up to the
configured one**, each on the model decoded from that level's file, and
records quality and size per level in `run.json`
([evals/RUN_SCHEMA.md](../evals/RUN_SCHEMA.md)).

| level | file (run dir) | what it stores | external code |
|---|---|---|---|
| `raw` | `checkpoints/raw.ckpt` | fp32 learned parameters: the candidate splats (means, scales, quaternions, opacities, colours), the instances (positions, rotations, scales, opacities, candidate assignment) and a small header. No optimizer state and no frozen DINO / LPIPS / discriminator weights. | none |
| `quantized` | `checkpoints/quantized.ckpt` | `raw` with fp16 geometry (splat means / scales / quaternions, instance positions / rotations / scales) and uint8 colour and opacity (per-tensor min–max). | none |
| `codec` (default) | `codec.omg` | `quantized` + OMG-style post-hoc coding: sub-vector quantization (SVQ, M=3, B=256) of the splat attributes, MPEG G-PCC for instance positions, SVQ (M=4, B=256) of instance quaternions, range coding of every integer stream, LZMA. | MPEG TMC13 (`tmc3`), constriction, scikit-learn |

Code: [scion/compression/levels.py](../scion/compression/levels.py)
(encode / decode / evaluate),
[quantize.py](../scion/compression/quantize.py) (`raw`, `quantized`),
[omg_posthoc.py](../scion/compression/omg_posthoc.py) and
[tmc13.py](../scion/compression/tmc13.py) (`codec`).

## Selecting a level

The level is the Hydra config group `compression`
([config/compression/](../config/compression)), default `codec`:

```bash
# Store and score raw + quantized only (no tmc3 needed):
python -m scion.train dataset=mipnerf360 scene=garden compression=quantized
# Codec options are the EncodeOptions fields in config/compression/codec.yaml:
python -m scion.train dataset=mipnerf360 scene=garden compression.tmc13_quant_step=1e-4
# The paper worker takes the same overrides:
python scripts/train_eval_render_one.py ... --override compression=quantized
# Score one level of a finished run (its artifact, decoded):
python -m scion.eval --ckpt <run_dir> --level quantized
```

Training itself does not depend on the level. `scion.engine.run`
(used by both the training CLI and `scripts/train_eval_render_one.py`) checks
the level's dependencies before training starts, and after training writes
each level's file and scores it with `levels.evaluate_levels`.

## Sizes

Records store bytes: `size_bytes` is the byte length of the level's file,
the file a user downloads. Tables show **MB (1 MB = 2^20 bytes)**
(`scion.util.units.MiB`, the convention of OMG and HAC++).
Bits per Gaussian is `size_bytes * 8 / n_gaussians`
(`scion.util.units.bits_per_gaussian`), where `n_gaussians` counts
the rendered world-space Gaussians.

## No silent substitutes

At level `codec` a missing `tmc3` binary, `constriction` or scikit-learn
raises an error that names `compression=quantized` as the way to skip the
level (for `python -m scion.eval`, pass `--level quantized` instead), and a `tmc3`
reconstruction that does not match the input grid raises too; nothing falls
back to a different algorithm under the codec name.
`run.json` records what was used: `codec.options`,
`codec.position_format_used`, `codec.tmc13_version`, and the
`constriction` / `scikit_learn` versions.

## Installing `tmc3`

```bash
bash scripts/setup/build_tmc13.sh   # prints the binary path
```

The script clones MPEG-PCC-TMC13 at the pinned `release-v23.0-rc2`
(`a3d15c5`, the version OMG and HAC++ use) and builds `tmc3` with CMake into
`${XDG_CACHE_HOME:-~/.cache}/scion/mpeg-pcc-tmc13`. It does not
rebuild if a binary is already found, but that binary must report the
pinned version or the script exits non-zero. Lookup order, shared with the
Python wrapper: `$TMC13_BINARY` (if set, nothing else is tried), the build
directory, `tmc3` on `PATH`.

## How the codec works

1. **SVQ of the splat-codebook buffers** (`splat_means`, `splat_scales`,
   `splat_quats`, `splat_features_dc`, `splat_opacities`). Each row is
   split into M sub-vectors, each quantized against ≤ B k-means codewords
   (scikit-learn `MiniBatchKMeans`, fixed seed); codebooks are stored fp16.
   Tensors with fewer than `min_rows_for_svq` rows pass through.
2. **Instance positions via MPEG G-PCC**: octree coding by `tmc3` on a grid
   of `tmc13_quant_step` (1e-3 world units). The decoder's octree walk
   reorders the points, so every per-instance tensor is permuted to that
   order at encode time and no permutation is stored.
3. **SVQ of instance quaternions** (M=4, i.e. scalar VQ per component).
4. **Range coding** (constriction) of SVQ indices, `instance_assignments`
   and `splat_candidate_id`.
5. **LZMA** over the whole container. The container is a JSON header plus
   raw buffers, so loading a file never executes code.

`position_format: fp16` replaces step 2 with a pure-Python Morton sort +
per-axis delta coding; it is recorded as such in
`codec.position_format_used`.

### Caveats

- **Scene-specific quant step.** `tmc13_quant_step` is in world units; the
  wrapper raises if the scene extent overflows G-PCC's grid.
- **The permutation is implicit.** It relies on `tmc3`'s octree walk being
  reproducible for the same integer grid (verified empirically, not
  specified by MPEG). Decoded per-instance buffers are consistent with
  each other but not in the original index order.

## Manual use

```bash
# Re-encode a quantized checkpoint with other codec options:
python scripts/compress_omg_posthoc.py \
    --input <run_dir>/checkpoints/quantized.ckpt --output codec_q1e-4.omg \
    --tmc13-quant-step 1e-4 --decode-verify
# Render and score a bundle:
python -m scion.eval --ckpt <run_dir> --level codec --artifact codec_q1e-4.omg
```

## Credits

- **OMG** — Lee, Ko & Park, *Optimized Minimal 3D Gaussian Splatting*,
  2025 ([github.com/maincold2/OMG](https://github.com/maincold2/OMG)). The
  codec level follows its design (SVQ of attributes, G-PCC for positions);
  `omg_posthoc.py` re-implements it for SCION's codebook layout and copies
  no OMG code.
- **MPEG-PCC-TMC13** — the G-PCC reference software
  ([github.com/MPEGGroup/mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)),
  BSD licence, pinned to `release-v23.0-rc2` (`a3d15c5`); run as a
  subprocess, not distributed here.
- **HAC++** — Chen et al.
  ([github.com/YihangChen-ee/HAC-plus](https://github.com/YihangChen-ee/HAC-plus)):
  the PLY-in / `tmc3` / PLY-out pattern in `tmc13.py` follows its gpcc
  utils, which OMG also uses.
- **constriction** ([github.com/bamler-lab/constriction](https://github.com/bamler-lab/constriction))
  — range coding.
- **scikit-learn** — k-means for SVQ.
