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


def _tile_2x2(frames: np.ndarray) -> np.ndarray:
    """(T, V=3, H, W, C) uint8 -> (T, 2H, 2W, C): [v0|v2] / [v1|black]."""
    t, v, h, w, c = frames.shape
    grid = np.zeros((t, 2 * h, 2 * w, c), dtype=np.uint8)
    grid[:, :h, :w] = frames[:, 0]
    grid[:, :h, w:] = frames[:, 2 if v > 2 else v - 1]
    grid[:, h:, :w] = frames[:, 1 if v > 1 else 0]
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
        meta = {
            "statistics": {
                "state": {"state": s["state.state"]},
                "action": {"action": s["action.action"]},
            },
            "modalities": {
                "video": {}, "state": {"state": {"shape": [self.source.state_dim]}},
                "action": {"action": {"shape": [self.source.action_dim]}},
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
        images = _tile_2x2(resized)  # (T, 2h, 2w, C) uint8

        action = np.zeros((_NUM_ACTION, _MAX_ACTION_DIM), dtype=np.float32)
        action_mask = np.zeros_like(action, dtype=bool)
        action[:, : self.source.action_dim] = s["action"]
        action_mask[:, : self.source.action_dim] = True

        state = np.zeros((1, _MAX_STATE_DIM), dtype=np.float32)
        state_mask = np.zeros_like(state, dtype=bool)
        state[0, : self.source.state_dim] = s["state"]
        state_mask[0, : self.source.state_dim] = True

        return {
            "images": images,
            "text": s["prompt"],
            "text_negative": (
                "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,"
                "低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,"
                "毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走"
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
