"""Minimal full-image dataloader.

Each step:

* picks a camera index (random or sequential),
* lazy-loads the image,
* returns ``{"camera", "image", "idx"}``.

No ray-bundle abstraction — gsplat consumes full images.
"""
from __future__ import annotations

import random
from typing import Iterator, Literal, Optional

import torch

from scion.camera import Camera
from scion.data.scene import Scene


class FullImageLoader:
    """Infinite full-image iterator over a :class:`Scene` split.

    Parameters
    ----------
    scene
        The scene to iterate over.
    split
        ``"train"`` or ``"eval"`` — picks which index list to use.
    shuffle
        Reshuffle each epoch when True. Ignored for ``next_eval``.
    device
        Device to move the camera + image to at yield time.
    seed
        RNG seed for the shuffle; ``None`` uses the global random state.
    """

    def __init__(
        self,
        scene: Scene,
        split: Literal["train", "eval"] = "train",
        shuffle: bool = True,
        device: str = "cuda",
        seed: Optional[int] = None,
    ):
        self.scene = scene
        self.split = split
        self.shuffle = shuffle
        self.device = device
        self._indices: list[int] = (
            list(scene.train_indices) if split == "train" else list(scene.eval_indices)
        )
        self._rng = random.Random(seed)
        self._order: list[int] = []  # the current epoch's camera order
        self._pos = 0                # next position in it
        # uint8 images on ``device``, filled on first use.
        self._images: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[dict]:
        while True:
            if self._pos >= len(self._order):
                # A fresh shuffle of the split per epoch from one seeded RNG
                # (seed = machine.seed).
                self._order = list(self._indices)
                if self.shuffle:
                    self._rng.shuffle(self._order)
                self._pos = 0
            self._pos += 1
            yield self._build_batch(self._order[self._pos - 1])

    def state_dict(self) -> dict:
        """Shuffle state (epoch order, position, RNG): a loader given it by
        :meth:`load_state_dict` yields the same cameras from here on."""
        return {"order": list(self._order), "pos": self._pos, "rng": self._rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        self._order, self._pos = list(state["order"]), int(state["pos"])
        self._rng.setstate(state["rng"])

    def next_eval(self, eval_idx: int) -> dict:
        """Deterministic single-image fetch by *eval-list* position.

        ``eval_idx=0`` returns the first scene-level eval image,
        ``eval_idx=1`` the second, etc.
        """
        return self._build_batch(self.scene.eval_indices[eval_idx])

    def _build_batch(self, idx: int) -> dict:
        scene = self.scene
        cam: Camera = scene.cameras[idx].to(self.device)
        if idx not in self._images:
            self._images[idx] = scene.load_image_uint8(idx).to(self.device)
        batch: dict = {"camera": cam, "image": self._images[idx], "idx": idx}
        return batch
