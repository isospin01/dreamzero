"""Train DreamZero on RoboCasa365 through OmniRobot's PRODUCTION lerobot-v3 data path.

Design (replaces the earlier bespoke ``RobocasaWindowSource`` bridge)
--------------------------------------------------------------------
Nothing about the data path is re-implemented here. This module wires up exactly
the objects ``OmniRobot/src/unified/data/lerobot/factory.py`` wires up for a
``configs/train/*.py`` ``DatasetEntry``::

    ShardedV3SingleDataset(dataset_path, embodiment_tag, modality_configs,
                           columns=topo.columns, adapter=topo.adapter,
                           target_fps=topo.target_fps, ...)
        -> ShardedLeRobotV3Mixture([single], [1.0], processor, batch_size=...)

and supplies a MINIMAL processor-like object (``_MinimalProcessor``) whose
transform chain is OmniRobot's own::

    FlattenVLAStep            # VLAStepData -> flat {video.<role>, state.<k>, action.<k>}
    QuantileNormalize("state")   # q01/q99 -> [-1, 1]   (OmniRobot's transform, verbatim)
    QuantileNormalize("action")  #     ""
    ConcatStateAction("state")   # per-key -> one vector, canonical key order
    ConcatStateAction("action")
    _ToDreamZeroSample           # the ONE adapter that lives on this side

The seam is therefore ``VLAStepData`` (canonical OmniRobot loader output) and the
normalization arithmetic stays on OmniRobot's side, driven by the stats the
mixture merges and injects via ``processor.set_statistics``. The values this
dataset feeds DreamZero are bit-identical to what the pi05 runs see up to the
model-specific finalization below. Statistics resolve through OmniRobot's stats
store (``assets/stats/<dataset>/stats.json``) automatically — nothing is passed in.

What ``_ToDreamZeroSample`` (and only it) is responsible for
-----------------------------------------------------------
* per-view stretch resize to ``image_hw`` (cv2 INTER_LINEAR). Same GEOMETRY as
  DreamZero's own ``VideoResize(interpolation=linear)`` at
  ``image_resolution_height/width`` — a plain stretch, deliberately NOT
  OmniRobot's letterbox ``VideoResizeWithPad``. Two deliberate differences from
  the DROID recipe's train chain: no antialias (their ``VideoResize`` defaults to
  the torchvision backend with ``antialias=True``) and no augmentation at all
  (their DROID chain adds a random 0.95 crop + color jitter). Augmentation is
  omitted so this baseline sees the same pixels our pi05 runs do;
* the DROID 2x2 tile: wrist across the full top row (width-doubled), left/right
  exteriors bottom-left/bottom-right — the layout ``DreamTransform._prepare_video``
  builds for ``EmbodimentTag.OXE_DROID`` and that the collate() prompt preamble
  describes in English;
* pad state/action to ``max_state_dim`` / ``max_action_dim`` + boolean masks,
  clip the (already normalized) action to [-1, 1] for the action head's assert,
  and mask out LeRobot's repeated-last-step action padding;
* the 15-key dict ``DreamTransform.apply_single`` produces, so this dataset is a
  drop-in for a ``dataset.transforms(step)`` output feeding ``collate()``.

Host-trainer contract (verified against this checkout)
------------------------------------------------------
``groot/vla/experiment/base.py``:

* ``BaseExperiment.__init__`` asserts ``train_dataset.merged_metadata is not None``
  and json-dumps ``{k: v.model_dump(mode="json")}`` -> we build one real
  ``DatasetMetadata`` (pydantic) per embodiment tag from the mixture's merged stats.
* ``create_trainer`` calls ``len(trainer.get_train_dataloader())`` and
  ``len(trainer.train_dataset)`` -> ``__len__`` is REQUIRED (both on the vanilla
  and on the custom path).
* ``BaseTrainer.get_train_dataloader`` only takes its bespoke
  (accelerate-bypassing) branch for ``ShardedLeRobotMixtureDataset``; anything
  else falls through to ``transformers.Trainer.get_train_dataloader``.

TRADEOFF — why this is a map-style ``Dataset`` over an infinite IterableDataset
-------------------------------------------------------------------------------
``ShardedLeRobotV3Mixture`` is an ``IterableDataset`` that partitions its shard
schedule across ``rank x dataloader-worker`` slots itself. Handing that object to
the trainer directly would take the vanilla HF path, where
``accelerator.prepare(DataLoader)`` defaults ``dispatch_batches=True`` for an
IterableDataset: only rank 0 would decode video and broadcast slices to every
other rank (a hard throughput ceiling for 3-view video), and forcing
``dispatch_batches=false`` instead wraps it in accelerate's ``IterableDatasetShard``,
where EVERY rank decodes the full stream and keeps 1/world of it. Neither is
acceptable, and this class is deliberately not a subclass of the baseline's
``ShardedLeRobotMixtureDataset`` just to trip an ``isinstance`` check.

So this is an honest iterable-to-map wrapper:

* ``__getitem__(index)`` IGNORES ``index`` and returns the next sample of this
  worker's slice of the infinite stream. The sampler is therefore decorative —
  it only controls HOW MANY items each rank pulls per epoch, which is all that
  matters here (the mixture, not the sampler, decides what data a slot sees).
* ``__len__`` is the NOMINAL ``steps_per_epoch``; the underlying stream is
  infinite, so it defines an epoch boundary rather than a dataset size. Runs are
  governed by ``max_steps`` (the experiment asserts ``max_steps > 0``). Keep it
  comfortably larger than one epoch of optimizer steps x global batch.
* Sharding is still correct and disjoint: each DataLoader worker builds its own
  iterator, and the mixture's ``_slot()`` resolves ``rank * num_workers +
  worker_id`` out of ``world_size * num_workers``.
* Batch/shard alignment: the mixture only emits complete groups of its own
  ``batch_size`` from a single shard, so pass ``batch_size=per_device_train_batch_size``
  (the yaml does). Dropping padded windows (below) can straddle that boundary;
  harmless here because there is exactly one dataset (uniform view count/shapes).
* Resume: ``set_epoch(e)`` (reached via ``BaseSampler.set_epoch`` ->
  ``data_source.set_epoch``) advances the mixture's ``epoch_offset`` and restarts
  the stream, so a new epoch never replays the previous one. For a mid-run resume
  set ``epoch_offset`` explicitly and run with ``ignore_data_skip=true``; the HF
  batch-skipping path would otherwise decode and discard thousands of windows.

Video padding (the previous file's KNOWN LIMITATION — now fixed)
----------------------------------------------------------------
``VLAStepData.pad_masks`` carries ``video.<key>`` / ``action.<key>`` ``is_pad``
flags from the v3 loader, and ``FlattenVLAStep`` forwards them as ``_video_pad`` /
``_action_pad``. Action padding is masked out of the loss via ``action_mask``.
DreamTransform's output contract has no video-pad slot, so a window whose video
runs past the end of the episode would feed LeRobot's repeated-last-frame padding
to the dynamics loss as real future video; with ``drop_padded_video_windows=True``
(default) those windows are skipped instead. The cost is a mild recency bias
(the last ``(num_video_frames-1) * stride`` steps of every episode are never an
anchor) and a broken per-batch single-shard guarantee, both preferable to
training the world model on frozen frames.

Chunking (K=1 chunk, matching the DROID recipe's CLI overrides): video T=9 frames
at stride 3 (1.2 s @ 20 fps), action (24, 12 -> pad 32), state (1, 16 -> pad 64).
``embodiment_id`` 17 = oxe_droid, the closest single-arm projector slot.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Iterator, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# The English negative prompt DreamTransform.apply_single hardcodes.
_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, "
    "artwork, painting, image, still, grayscale, dull, worst quality, low quality, "
    "JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, "
    "disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, "
    "three legs, many people in the background, walking backwards."
)

# Private key carrying the video-pad verdict from the transform chain to
# __getitem__. ALWAYS popped there, so the sample handed to collate() has exactly
# the 15 keys DreamTransform.apply_single produces.
_PAD_FLAG = "_omni_video_is_pad"

# DROID tile slots, in the order _tile_droid expects:
#   (bottom-left exterior, bottom-right exterior, top wrist).
# These are OmniRobot ViewRole names (unified/data/view_roles.py); the
# robocasa_panda_omron topology maps
#   agentview_left -> external_0, agentview_right -> external_1,
#   eye_in_hand    -> left_wrist.
_DEFAULT_VIEW_ORDER = ("external_0", "external_1", "left_wrist")


def _tile_droid(frames: np.ndarray) -> np.ndarray:
    """(T, V=3, H, W, C) uint8 -> (T, 2H, 2W, C) in the OXE_DROID layout the id-17
    checkpoint and collate()'s prompt preamble expect: wrist across the full top row
    (width-doubled), exteriors bottom-left / bottom-right.
    View order: 0 = left exterior, 1 = right exterior, 2 = wrist."""
    t, v, h, w, c = frames.shape
    assert v == 3, f"DROID tiling needs exactly 3 views, got {v}"
    grid = np.zeros((t, 2 * h, 2 * w, c), dtype=np.uint8)
    grid[:, :h, :] = np.repeat(frames[:, 2], 2, axis=-2)  # wrist, width-doubled
    grid[:, h:, :w] = frames[:, 0]  # left exterior
    grid[:, h:, w:] = frames[:, 1]  # right exterior
    return grid


class _ToDreamZeroSample:
    """Flat, NORMALIZED OmniRobot sample dict -> DreamZero's per-sample dict.

    Duck-types OmniRobot's ``ModalityTransform`` (``__call__`` / ``set_statistics``
    / ``train`` / ``eval``) so it can be the last link of a
    ``ComposedModalityTransform`` without importing ``unified`` at module import
    time. It is NOT an ``InvertibleModalityTransform``: the chain's ``unapply``
    (decode) skips it, exactly like OmniRobot's own model finalizers.

    Input keys (post FlattenVLAStep + QuantileNormalize + ConcatStateAction):
    ``video.<role>`` (T,H,W,C) uint8, ``state`` (1,Ds), ``action`` (H,Da),
    ``text``, ``embodiment``, optional ``_video_pad`` / ``_action_pad``.
    """

    def __init__(
        self,
        *,
        image_hw: tuple[int, int],
        view_order: Sequence[str],
        max_action_dim: int,
        max_state_dim: int,
        embodiment_id: int,
        negative_prompt: str = _NEGATIVE_PROMPT,
    ) -> None:
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.view_order = tuple(view_order)
        self.max_action_dim = int(max_action_dim)
        self.max_state_dim = int(max_state_dim)
        self.embodiment_id = int(embodiment_id)
        self.negative_prompt = negative_prompt
        self.training = True

    # -- ModalityTransform surface (duck-typed) --------------------------- #
    def set_statistics(self, statistics: dict[str, Any]) -> None:
        return None

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.apply(data)

    # -- the actual conversion -------------------------------------------- #
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        import cv2  # opencv-python is a dreamzero dependency; keep the import local

        h, w = self.image_hw
        video_pad = data.get("_video_pad") or {}
        views: list[np.ndarray] = []
        window_is_padded = False
        for role in self.view_order:
            key = f"video.{role}"
            if key not in data:
                raise KeyError(
                    f"{type(self).__name__}: camera {key!r} missing; the sample has "
                    f"{sorted(k for k in data if isinstance(k, str) and k.startswith('video.'))}. "
                    "Check `view_order` against the embodiment topology's video_roles."
                )
            frames = np.asarray(data[key], dtype=np.uint8)  # (T, H, W, C)
            views.append(np.stack([cv2.resize(f, (w, h)) for f in frames]))
            pad = video_pad.get(role)
            if pad is not None and bool(np.any(pad)):
                window_is_padded = True
        images = _tile_droid(np.stack(views, axis=1))  # (T, 2h, 2w, C) uint8

        # --- action: normalized (H, Da) -> (H, max_action_dim) + mask ----- #
        act = np.asarray(data["action"], dtype=np.float32)
        if act.ndim != 2:
            raise ValueError(f"expected a 2-D (horizon, dim) action, got {act.shape}")
        n_act = act.shape[-1]
        if n_act > self.max_action_dim:
            raise ValueError(
                f"action width {n_act} exceeds max_action_dim {self.max_action_dim}"
            )
        action = np.zeros((act.shape[0], self.max_action_dim), dtype=np.float32)
        # The action head asserts actions land in [-1, 1]; clip AFTER the exact shared
        # q01/q99 normalization (DreamZero's native q99 pipeline is bounded the same way).
        action[:, :n_act] = np.clip(act, -1.0, 1.0)
        action_mask = np.zeros_like(action, dtype=bool)
        action_pad = data.get("_action_pad")
        if action_pad is None:
            action_mask[:, :n_act] = True
        else:  # mask real dims AND drop LeRobot's repeated-last-step padding
            action_mask[:, :n_act] = ~np.asarray(action_pad, dtype=bool).reshape(-1, 1)

        # --- state: normalized (T_s, Ds) -> (T_s, max_state_dim) + mask --- #
        st = np.asarray(data["state"], dtype=np.float32)
        if st.ndim == 1:
            st = st[None]
        n_state = st.shape[-1]
        if n_state > self.max_state_dim:
            raise ValueError(
                f"state width {n_state} exceeds max_state_dim {self.max_state_dim}"
            )
        state = np.zeros((st.shape[0], self.max_state_dim), dtype=np.float32)
        state[:, :n_state] = st
        state_mask = np.zeros_like(state, dtype=bool)
        state_mask[:, :n_state] = True

        return {
            "images": images,
            "text": data.get("text") or "",
            "text_negative": self.negative_prompt,
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
            "embodiment_id": self.embodiment_id,
            _PAD_FLAG: window_is_padded,
        }


class _MinimalProcessor:
    """The processor-like object ``ShardedLeRobotV3Mixture`` duck-types.

    The mixture needs exactly two things: ``train_chain`` / ``eval_chain`` (a
    callable applied to each ``VLAStepData``) and ``set_statistics(merged_stats)``,
    which it calls in ``__init__`` with the stats merged across the mixture. The
    DataLoader-side surface (``collator`` / ``get_modality_configs``) is only used
    by OmniRobot's own factory, not here — the baseline's own collator batches.
    """

    def __init__(self, chain: Any) -> None:
        self.train_chain = chain
        self.eval_chain = chain
        self.statistics: dict[str, Any] | None = None

    def set_statistics(self, statistics: dict[str, Any]) -> None:
        self.statistics = statistics
        self.train_chain.set_statistics(statistics)
        if self.eval_chain is not self.train_chain:
            self.eval_chain.set_statistics(statistics)


class OmniRobocasaDataset(torch.utils.data.Dataset):
    """RoboCasa365 (LeRobot v3) -> DreamZero samples via OmniRobot's v3 loader.

    Args:
        dataset_path: ``hf://datasets/<owner>/<name>`` or a local v3 root. Read
            straight off the Hub (only ``meta/`` is prefetched).
        embodiment: OmniRobot embodiment name; selects the topology (columns,
            adapter, camera roles, target_fps) and the statistics bucket.
        omnirobot_src: path to OmniRobot's ``src`` dir, prepended to ``sys.path``
            when set. Leave null if ``unified`` is already importable.
        action_horizon / num_video_frames: window geometry. Video delta indices are
            ``range(0, action_horizon + 1, action_horizon // (num_video_frames - 1))``,
            i.e. the video clip spans exactly the action chunk (OmniRobot's
            ``unified/models/dreamzero/modality.py`` convention).
        num_frame_per_block / num_action_per_block / num_state_per_block: when
            given, the same block geometry the action head asserts on, checked here
            so a mis-shaped config fails at dataset construction, not 30 min into
            the first forward.
        batch_size: MUST equal ``per_device_train_batch_size`` — it is the mixture's
            per-shard drop-tail group size.
        steps_per_epoch: nominal ``__len__`` (the stream is infinite).
    """

    def __init__(
        self,
        dataset_path: str = "hf://datasets/ember-lab-berkeley/robocasa365-pretrain-atomic",
        embodiment: str = "robocasa_panda_omron",
        omnirobot_src: str | None = None,
        *,
        action_horizon: int = 24,
        num_video_frames: int = 9,
        image_hw: tuple[int, int] = (176, 320),
        max_action_dim: int = 32,
        max_state_dim: int = 64,
        embodiment_id: int = 17,
        metadata_embodiment_tag: str = "oxe_droid",
        view_order: Sequence[str] = _DEFAULT_VIEW_ORDER,
        batch_size: int = 1,
        shard_size: int = 2000,
        seed: int = 42,
        shards_per_epoch: int | None = None,
        shard_sampling_rate: float = 1.0,
        discard_bad_trajectories: bool = True,
        video_backend: str = "torchcodec",
        drop_padded_video_windows: bool = True,
        steps_per_epoch: int = 1_000_000,
        epoch_offset: int = 0,
        num_frame_per_block: int | None = None,
        num_action_per_block: int | None = None,
        num_state_per_block: int = 1,
    ) -> None:
        if omnirobot_src and omnirobot_src not in sys.path:
            sys.path.insert(0, str(omnirobot_src))

        from unified.data.embodiment_topology import get_topology
        from unified.data.gr00t.embodiment_tags import EmbodimentTag
        from unified.data.gr00t.types import ModalityConfig
        from unified.data.lerobot.sharded_dataset import (
            ShardedLeRobotV3Mixture,
            ShardedV3SingleDataset,
        )
        from unified.data.transforms.adapter import FlattenVLAStep
        from unified.data.transforms.base import ComposedModalityTransform
        from unified.data.transforms.state_action import ConcatStateAction, QuantileNormalize

        video_deltas = self._video_deltas(action_horizon, num_video_frames)
        self._check_block_geometry(
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            num_frame_per_block=num_frame_per_block,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )

        tag = EmbodimentTag.resolve(embodiment)
        topo = get_topology(embodiment)
        # Guards against SILENTLY diverging from the OmniRobot runs: both of these
        # topology features change what "normalized action" means, and this chain
        # implements neither. robocasa_panda_omron sets neither.
        if topo.action_transform is not None:
            raise NotImplementedError(
                f"embodiment {embodiment!r} declares action_transform="
                f"{topo.action_transform!r}; this chain normalizes ABSOLUTE actions "
                "against statistics['action'] only. Add RelativeEEFAction (and the "
                "matching relative stats bucket) before QuantileNormalize first."
            )
        if topo.actions_are_observations:
            raise NotImplementedError(
                f"embodiment {embodiment!r} has actions_are_observations=True, which "
                "requires sampling action deltas [1..H] instead of [0..H-1]."
            )
        self.embodiment_value = tag.value
        self.metadata_embodiment_tag = metadata_embodiment_tag
        self.drop_padded_video_windows = bool(drop_padded_video_windows)
        # Steps the window reaches past its anchor (video and action agree by
        # construction: video_deltas[-1] == action_horizon).
        self.window_steps = int(video_deltas[-1])
        self.steps_per_epoch = int(steps_per_epoch)
        self.batch_size = int(batch_size)
        self._epoch_offset = int(epoch_offset)
        self._iter: Iterator[dict[str, Any]] | None = None
        self._skipped_padded = 0
        self._served = 0

        # Modality configs: the model-intrinsic half (how much to sample), composed
        # with the topology's data-intrinsic half (which keys exist) — the same
        # split OmniRobot's `models/<model>/modality.py` builders use.
        modality_config = {
            "video": ModalityConfig(
                delta_indices=video_deltas, modality_keys=list(topo.video_keys)
            ),
            "state": ModalityConfig(delta_indices=[0], modality_keys=list(topo.state_keys)),
            "action": ModalityConfig(
                delta_indices=list(range(action_horizon)), modality_keys=list(topo.action_keys)
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[topo.language_key]),
        }
        modality_configs = {tag.value: modality_config}
        video_roles = {tag.value: [int(r) for r in (topo.video_roles or [])]}

        chain = ComposedModalityTransform(
            [
                FlattenVLAStep(modality_configs, video_roles),
                QuantileNormalize("state"),
                QuantileNormalize("action"),
                ConcatStateAction("state", modality_configs),
                ConcatStateAction("action", modality_configs),
                _ToDreamZeroSample(
                    image_hw=image_hw,
                    view_order=view_order,
                    max_action_dim=max_action_dim,
                    max_state_dim=max_state_dim,
                    embodiment_id=embodiment_id,
                ),
            ]
        )
        chain.train()
        self.processor = _MinimalProcessor(chain)

        single = ShardedV3SingleDataset(
            dataset_path=dataset_path,
            embodiment_tag=tag,
            modality_configs=modality_config,
            columns=topo.columns,       # name-derived slice spec from the topology
            adapter=topo.adapter,       # dataset-specific conversion; None => identity slice
            target_fps=topo.target_fps, # None = native rate
            shard_size=shard_size,
            seed=seed,
            video_backend=video_backend,
            discard_bad_trajectories=discard_bad_trajectories,
        )
        # Merges statistics and injects them into the processor in __init__.
        self.mixture = ShardedLeRobotV3Mixture(
            [single],
            [1.0],
            self.processor,
            batch_size=self.batch_size,
            seed=seed,
            shards_per_epoch=shards_per_epoch,
            training=True,
            shard_sampling_rate=shard_sampling_rate,
        )
        self.mixture.epoch_offset = self._epoch_offset

        self.num_frames_available = single.num_frames
        self.merged_metadata = self._build_metadata(self.mixture.global_stats)
        logger.info(
            "OmniRobocasaDataset: %s [%s] %d usable frames over %d shards; "
            "video deltas %s, action horizon %d, nominal epoch %d",
            dataset_path, tag.value, single.num_frames, len(single.shards),
            video_deltas, action_horizon, self.steps_per_epoch,
        )

    # ------------------------------------------------------------------ #
    # geometry                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _video_deltas(action_horizon: int, num_video_frames: int) -> list[int]:
        if num_video_frames < 2:
            raise ValueError(f"num_video_frames must be >= 2, got {num_video_frames}")
        if (num_video_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_video_frames must be 4k+1 for the Wan VAE's temporal stride, "
                f"got {num_video_frames}"
            )
        span = num_video_frames - 1
        if action_horizon % span != 0:
            raise ValueError(
                f"action_horizon ({action_horizon}) must be divisible by "
                f"num_video_frames - 1 ({span}) so the video clip spans the action chunk"
            )
        stride = action_horizon // span
        return list(range(0, action_horizon + 1, stride))

    @staticmethod
    def _check_block_geometry(
        *,
        num_video_frames: int,
        action_horizon: int,
        num_frame_per_block: int | None,
        num_action_per_block: int | None,
        num_state_per_block: int,
    ) -> None:
        """Mirror the two asserts in ``action_head/wan_flow_matching_action_tf.py``.

        Latent frames after the Wan VAE's 4x temporal downsample:
        ``L = (T - 1) // 4 + 1``; the head requires
        ``action_horizon / (L - 1) == num_action_per_block // num_frame_per_block``
        and ``(L - 1) / state_horizon == num_frame_per_block // num_state_per_block``
        with ``state_horizon == 1`` here.
        """
        if num_frame_per_block is None or num_action_per_block is None:
            return
        latent_frames = (num_video_frames - 1) // 4 + 1
        blocks = latent_frames - 1
        if blocks <= 0:
            raise ValueError(f"num_video_frames={num_video_frames} leaves no latent blocks")
        want = num_action_per_block // num_frame_per_block
        got = action_horizon / blocks
        if got != want:
            raise ValueError(
                f"block geometry mismatch: action_horizon/{blocks} latent blocks = {got}, "
                f"but num_action_per_block // num_frame_per_block = {want} "
                f"(num_frames={num_video_frames}, action_horizon={action_horizon}, "
                f"num_action_per_block={num_action_per_block}, "
                f"num_frame_per_block={num_frame_per_block})"
            )
        want_state = num_frame_per_block // num_state_per_block
        if blocks / 1 != want_state:  # state_horizon == 1 (state delta_indices == [0])
            raise ValueError(
                f"state block geometry mismatch: {blocks} latent blocks per state token, "
                f"but num_frame_per_block // num_state_per_block = {want_state}"
            )

    # ------------------------------------------------------------------ #
    # trainer contract                                                   #
    # ------------------------------------------------------------------ #
    def _build_metadata(self, stats: dict[str, Any]) -> dict[str, Any]:
        """The pydantic metadata BaseExperiment asserts on and dumps to
        ``experiment_cfg/metadata.json`` (read back by the inference policy for
        un-normalization). Built from the SAME merged statistics the chain
        normalizes with, so metadata.json can never drift from training."""
        from groot.vla.data.schema.lerobot import DatasetMetadata

        emb = stats[self.embodiment_value]
        fields = ("min", "max", "mean", "std", "q01", "q99")

        def _values(bucket: dict[str, Any]) -> dict[str, dict[str, Any]]:
            return {k: {f: v[f] for f in fields} for k, v in bucket.items()}

        def _shapes(bucket: dict[str, Any]) -> dict[str, dict[str, Any]]:
            return {
                k: {
                    "shape": [len(np.asarray(v["q01"]).reshape(-1))],
                    "absolute": True,
                    "continuous": True,
                }
                for k, v in bucket.items()
            }

        meta = {
            "statistics": {"state": _values(emb["state"]), "action": _values(emb["action"])},
            "modalities": {
                # Video metadata is unused downstream (the policy reads statistics);
                # the real per-camera resolution is only known after decode.
                "video": {},
                "state": _shapes(emb["state"]),
                "action": _shapes(emb["action"]),
            },
            "embodiment_tag": self.metadata_embodiment_tag,
        }
        return {self.metadata_embodiment_tag: DatasetMetadata.model_validate(meta)}

    def __len__(self) -> int:
        """NOMINAL epoch size — the underlying stream is infinite (see module docstring).
        Required: ``create_trainer`` calls ``len(dataloader)`` and ``len(train_dataset)``."""
        return self.steps_per_epoch

    def __str__(self) -> str:
        return (
            f"OmniRobocasaDataset(embodiment={self.embodiment_value!r}, "
            f"frames={self.num_frames_available}, nominal_len={self.steps_per_epoch}, "
            f"batch_size={self.batch_size}, epoch_offset={self.mixture.epoch_offset})"
        )

    __repr__ = __str__

    def set_epoch(self, epoch: int) -> None:
        """Called by ``BaseSampler.set_epoch`` at every epoch boundary.

        Advances the mixture's ``epoch_offset`` (its shard schedule AND within-shard
        permutation are seeded on the epoch) and drops the current iterator, so epoch
        e+1 draws a fresh schedule instead of replaying epoch e."""
        target = self._epoch_offset + int(epoch)
        if target == getattr(self.mixture, "epoch_offset", 0):
            return
        self.mixture.epoch_offset = target
        self._iter = None

    # ------------------------------------------------------------------ #
    # sample access                                                      #
    # ------------------------------------------------------------------ #
    def _iterator(self) -> Iterator[dict[str, Any]]:
        if self._iter is None:
            # Built lazily so each DataLoader worker process gets its own slot
            # (the mixture resolves rank x worker inside __iter__).
            self._iter = iter(self.mixture)
        return self._iter

    def __getitem__(self, index: int) -> dict[str, Any]:
        """``index`` is IGNORED — see the tradeoff section of the module docstring."""
        it = self._iterator()
        skipped = 0
        while True:
            try:
                sample = next(it)
            except StopIteration as e:  # the mixture is infinite by construction
                raise RuntimeError(
                    "OmniRobocasaDataset: the OmniRobot mixture stopped yielding; "
                    "this should be unreachable (ShardedLeRobotV3Mixture iterates "
                    "itertools.count(epoch))."
                ) from e
            padded = bool(sample.pop(_PAD_FLAG, False))
            if not padded or not self.drop_padded_video_windows:
                self._served += 1
                return sample
            skipped += 1
            self._skipped_padded += 1
            if skipped >= 256:
                logger.warning(
                    "OmniRobocasaDataset: 256 consecutive windows with padded video; "
                    "serving a padded one to keep the pipeline moving (are the episodes "
                    "shorter than the %d-step window?).",
                    self.window_steps,
                )
                self._served += 1
                return sample
