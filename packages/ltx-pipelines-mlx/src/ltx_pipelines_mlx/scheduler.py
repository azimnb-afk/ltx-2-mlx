"""LTX-2 sigma schedules.

`ltx2_schedule` is a thin wrapper over `mlx_arsenal.diffusion.dynamic_shift_schedule`
that preserves LTX's original keyword name (``steps``) and default ``num_tokens``.
The predefined LTX-specific tables (DISTILLED_SIGMAS, STAGE_2_SIGMAS) and the
LTX-only helpers (get_sigma_schedule, sigma_to_timestep) stay local.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_arsenal.diffusion import dynamic_shift_schedule

_MAX_SHIFT_ANCHOR = 4096


def ltx2_schedule(
    steps: int,
    num_tokens: int = _MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """LTX-2 token-count-adaptive flow-matching sigma schedule."""
    return dynamic_shift_schedule(
        steps,
        num_tokens=num_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
        stretch=stretch,
        terminal=terminal,
    )


__all__ = [
    "DISTILLED_SIGMAS",
    "STAGE_2_SIGMAS",
    "STAGE_2_SIGMAS_LTX25",
    "resolve_stage2_sigmas",
    "get_sigma_schedule",
    "ltx2_schedule",
    "sigma_to_timestep",
]

# Predefined sigma schedule for 8-step distilled model.
# 9 values = 8 steps (iterate consecutive pairs: sigmas[i], sigmas[i+1]).
DISTILLED_SIGMAS: list[float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# Sigma schedule for stage 2 refinement (two-stage pipeline).
# 4 values = 3 steps.
#
# This is the LTX-2.3 schedule, and it is exactly the tail of DISTILLED_SIGMAS
# (indices 5..8) — stage 2 picks up where stage 1's last few steps were.
STAGE_2_SIGMAS: list[float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# LTX-2.5's stage-2 schedule, read verbatim from the official ComfyUI template
# `video_ltx2_5_t2v.json` (node 395, `ManualSigmas`): "0.85, 0.7250, 0.4219, 0.0".
#
# **Only the FIRST value differs from 2.3** — 0.85 where we use 0.909375. The
# other two are the same numbers the template prints at four decimal places
# (0.4219 is 0.421875 rounded for display, not a different sigma), which is what
# makes the first value's change deliberate rather than a transcription artifact:
# whoever wrote the template took 2.3's tail and moved one number.
#
# It matters because it is a REFINE pass. Starting stage 2 at 0.909375 re-noises
# the upscaled latent harder than the vendor does, so the refine deviates further
# from the stage-1 result. We shipped the 2.3 value on 2.5 until 2026-08-12.
#
# Stage 1 needs no such entry: our DISTILLED_SIGMAS is byte-identical to the same
# template's node 404, all nine values including the five inside the top 2.5% of
# the noise range that no naive shifted schedule reproduces.
STAGE_2_SIGMAS_LTX25: list[float] = [
    0.85,
    0.725,
    0.421875,
    0.0,
]


def resolve_stage2_sigmas(model_version: tuple[int, ...], stage2_steps: int | None = None) -> list[float]:
    """Stage-2 schedule for a checkpoint generation, truncated as the caller asks.

    Args:
        model_version: The checkpoint's generation, e.g. ``(2, 5)``. Anything
            below ``(2, 5)`` — including the empty tuple an unreadable
            checkpoint yields — gets the 2.3 schedule, so an unknown checkpoint
            keeps the older behaviour instead of opting into a newer schedule.
        stage2_steps: Number of steps; ``None``/0 means the full schedule.

    Returns:
        The sigma list. **Truncation keeps the pre-existing semantics exactly**,
        including the wart that a truncated schedule no longer ends at 0.0 and
        therefore leaves residual noise. That behaviour predates this function
        and is not this function's to change.
    """
    sigmas = STAGE_2_SIGMAS_LTX25 if tuple(model_version) >= (2, 5) else STAGE_2_SIGMAS
    return sigmas[: stage2_steps + 1] if stage2_steps else sigmas


def get_sigma_schedule(
    schedule_name: str = "distilled",
    num_steps: int | None = None,
) -> list[float]:
    """Get a sigma schedule by name.

    Args:
        schedule_name: "distilled" or "stage_2".
        num_steps: Optional number of steps (truncates schedule).

    Returns:
        List of sigma values.
    """
    if schedule_name == "distilled":
        sigmas = DISTILLED_SIGMAS
    elif schedule_name == "stage_2":
        sigmas = STAGE_2_SIGMAS
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    if num_steps is not None:
        sigmas = sigmas[:num_steps]
    return sigmas


def sigma_to_timestep(sigma: float) -> mx.array:
    """Convert sigma to timestep array.

    Args:
        sigma: Noise level.

    Returns:
        Timestep as (1,) array.
    """
    return mx.array([sigma], dtype=mx.bfloat16)
