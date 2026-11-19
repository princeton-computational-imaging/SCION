# Third-party notices

This project is licensed under the Apache License 2.0 (see
[LICENSE](LICENSE)). It incorporates or derives from the following
third-party work.

## Ported code (Apache-2.0)

The files below contain code ported or adapted from
[nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
(Copyright 2022 the nerfstudio team, Apache-2.0). Each file carries a
notice header stating what was changed.

| File | Upstream origin |
|---|---|
| `scion/data/colmap.py` | COLMAP parsing / pose-convention logic following `nerfstudio.data.dataparsers.nerfstudio_dataparser` and `nerfstudio.process_data.colmap_utils` |
| `scion/data/blender.py` | Blender-format parsing following `nerfstudio.data.dataparsers.blender_dataparser` |

## Library dependencies (not vendored)

- [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0) —
  the rasterization backend; called, not copied.
- [pytorch-msssim](https://github.com/VainF/pytorch-msssim) (MIT) —
  SSIM loss (`scion/engine/image_loss.py` wraps the installed
  package; no code is copied).
- [lpips](https://github.com/richzhang/PerceptualSimilarity)
  (BSD-2-Clause) — perceptual metric.
- [MPEG-PCC-TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)
  (BSD licence, Copyright ISO/IEC) — the G-PCC reference codec. Its `tmc3`
  binary, pinned to `release-v23.0-rc2` (`a3d15c5`), is built by
  `scripts/setup/build_tmc13.sh` and invoked as a subprocess by
  `scion/compression/tmc13.py` for compression level `codec`;
  not distributed with this repository.
- [constriction](https://github.com/bamler-lab/constriction)
  (MIT OR Apache-2.0 OR BSL-1.0) — range coding in compression level
  `codec`.
- [scikit-learn](https://scikit-learn.org) (BSD-3-Clause) — k-means for
  the sub-vector quantization in compression level `codec`.
- [COLMAP](https://colmap.github.io) (BSD-3-Clause), run as an external
  program (in the pixi environment) — triangulates the seed points of
  Synthetic-NeRF scenes (`scion.data.synthetic`) and reconstructs your
  own scenes (`scripts/colmap_scene.sh`).
- [nerfview](https://github.com/nerfstudio-project/nerfview) (Apache-2.0)
  and [viser](https://github.com/nerfstudio-project/viser) (MIT) —
  optional `[viewer]` extra: the live viewer (`trainer.vis=viewer`).
- [DINOv3](https://github.com/facebookresearch/dinov3) weights (Meta AI;
  `dinov3_vitl16` by default) are downloaded on first use, by training or
  `scripts/preprocess_dino.py`, under Meta's DINOv3 licence terms; they are
  not distributed here.

## Provenance

No code is taken from the original INRIA 3D Gaussian Splatting reference
implementation, which is under a non-commercial research-only licence
incompatible with this repository. The `codec` compression level
(`scion/compression/omg_posthoc.py`) follows the design of OMG
(Lee, Ko & Park, *Optimized Minimal 3D Gaussian Splatting*, 2025,
[github.com/maincold2/OMG](https://github.com/maincold2/OMG)) and is
implemented from the paper, not from the OMG repository. The `tmc3`
subprocess pattern in `scion/compression/tmc13.py` (write a PLY,
run the encoder / decoder, read the PLY back) follows the gpcc utils of
HAC++ ([github.com/YihangChen-ee/HAC-plus](https://github.com/YihangChen-ee/HAC-plus)),
which OMG also uses; HAC++ is under the Inria Gaussian-Splatting licence,
so only the approach is reused and none of its code is included. The OMG
*baseline numbers* in `evals/` were produced by running the upstream OMG
code unmodified in its own checkout.
