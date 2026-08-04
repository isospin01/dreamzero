"""OmniRobot data bridge: train DreamZero on RoboCasa365 via OmniRobot's loader.

Map-style dataset yielding the exact per-sample dict DreamTransform produces
(see groot/vla/model/dreamzero/transform/dreamzero_cotrain.py), but sourced from
OmniRobot's RobocasaWindowSource — same LeRobot-v3 data, same q01/q99->[-1,1]
normalization as the pi05 runs. Inject via hydra:

    train_dataset:
      _target_: groot.vla.data.dataset.omni_robocasa.OmniRobocasaDataset
      omnirobot_src: /app/code/src
      dataset_root: ...
      stats_path: ...

Chunking matches the DROID recipe with K=1 chunk: video T=9 frames at stride 3
(1.2 s @ 20 fps), action (24, 12->pad 32), state (1, 16->pad 64).
embodiment_id 17 (oxe_droid: closest single-arm layout; reuses its projector).
"""

from __future__ import annotations

import sys

import numpy as np
import torch

_NUM_ACTION = 24
_VIDEO_OFFSETS = tuple(range(0, 25, 3))  # 9 frames, T % 8 == 1
_MAX_ACTION_DIM = 32
_MAX_STATE_DIM = 64
_EMBODIMENT_ID = 17  # oxe_droid slot


def _tile_droid(frames: np.ndarray) -> np.ndarray:
    """(T, V=3, H, W, C) uint8 -> (T, 2H, 2W, C) in the OXE_DROID layout the id-17
    checkpoint and collate() preamble expect: wrist across the full top row
    (width-doubled), exteriors bottom-left / bottom-right.
    Mapping: eye_in_hand -> wrist, agentview_left/right -> exteriors."""
    t, v, h, w, c = frames.shape
    grid = np.zeros((t, 2 * h, 2 * w, c), dtype=np.uint8)
    grid[:, :h, :] = np.repeat(frames[:, 2], 2, axis=-2)  # wrist, width-doubled
    grid[:, h:, :w] = frames[:, 0]  # left exterior
    grid[:, h:, w:] = frames[:, 1]  # right exterior
    return grid


class OmniRobocasaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        omnirobot_src: str,
        dataset_root: str,
        stats_path: str,
        image_hw: tuple[int, int] = (176, 320),
    ):
        if omnirobot_src not in sys.path:
            sys.path.insert(0, omnirobot_src)
        from unified.data.baseline_bridge import RobocasaWindowSource

        self.source = RobocasaWindowSource(
            dataset_root=dataset_root,
            stats_path=stats_path,
            action_horizon=_NUM_ACTION,
            video_offsets=_VIDEO_OFFSETS,
        )
        self.image_hw = image_hw
        self.merged_metadata = self._build_metadata()

    # -- trainer contract -------------------------------------------------
    def _build_metadata(self):
        from groot.vla.data.schema.lerobot import DatasetMetadata

        s = self.source.raw_stats
        stat = lambda d: {k: v for k, v in d.items() if k != "count"}
        meta = {
            "statistics": {
                "state": {"state": stat(s["state.state"])},
                "action": {"action": stat(s["action.action"])},
            },
            "modalities": {
                "video": {},
                "state": {
                    "state": {
                        "shape": [self.source.state_dim],
                        "absolute": True, "continuous": True,
                    }
                },
                "action": {
                    "action": {
                        "shape": [self.source.action_dim],
                        "absolute": True, "continuous": True,
                    }
                },
            },
            "embodiment_tag": "oxe_droid",
        }
        try:
            return {"oxe_droid": DatasetMetadata.model_validate(meta)}
        except Exception:  # schema drift: trainer only json-dumps it
            return {"oxe_droid": DatasetMetadata.model_construct(**meta)}

    def __len__(self) -> int:
        return len(self.source)

    def __str__(self) -> str:
        return f"OmniRobocasaDataset(n={len(self)})"

    def __getitem__(self, index: int) -> dict:
        import cv2

        s = self.source[index]
        h, w = self.image_hw
        frames = s["video"]  # (T, V, H0, W0, C)
        t, v = frames.shape[:2]
        resized = np.stack(
            [
                np.stack([cv2.resize(frames[i, j], (w, h)) for j in range(v)])
                for i in range(t)
            ]
        )
        images = _tile_droid(resized)  # (T, 2h, 2w, C) uint8

        action = np.zeros((_NUM_ACTION, _MAX_ACTION_DIM), dtype=np.float32)
        action_mask = np.zeros_like(action, dtype=bool)
        # DreamZero's action head asserts actions in [-1,1]; clip AFTER the exact
        # shared normalization (their native q99 pipeline is bounded the same way).
        action[:, : self.source.action_dim] = np.clip(s["action"], -1.0, 1.0)
        # Mask real dims AND mask out lerobot's repeated-last-step padding.
        action_mask[:, : self.source.action_dim] = ~s["action_is_pad"][:, None]

        state = np.zeros((1, _MAX_STATE_DIM), dtype=np.float32)
        state_mask = np.zeros_like(state, dtype=bool)
        state[0, : self.source.state_dim] = s["state"]
        state_mask[0, : self.source.state_dim] = True

        return {
            "images": images,
            "text": s["prompt"],
            "text_negative": (
                "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, "
                "artwork, painting, image, still, grayscale, dull, worst quality, low quality, "
                "JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, "
                "disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, "
                "three legs, many people in the background, walking backwards."
            ),
            "state": state,
            "state_mask": state_mask,
            "action": action,
            "action_mask": action_mask,
            "lapa_action": np.zeros_like(action),
            "lapa_action_mask": np.zeros_like(action_mask),
            "segmentation_target": np.zeros((2,), dtype=np.float32),
            "segmentation_target_mask": np.zeros((1,), dtype=np.float32),
            "has_real_action": np.ones((), dtype=bool),
            "has_lapa_action": np.zeros((), dtype=bool),
            "is_cotrain_instance": np.zeros((), dtype=bool),
            "embodiment_id": _EMBODIMENT_ID,
        }
