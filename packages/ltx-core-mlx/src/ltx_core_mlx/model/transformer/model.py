"""LTXModel -- top-level DiT model for joint audio+video diffusion.

Ported from ltx-core/src/ltx_core/model/model.py

Top-level weight keys (after stripping ``transformer.`` prefix):
    adaln_single, audio_adaln_single         -- 9-param timestep AdaLN
    prompt_adaln_single, audio_prompt_adaln_single  -- 2-param text cross-attn
    av_ca_video_scale_shift_adaln_single     -- 4-param AV cross-attn video
    av_ca_audio_scale_shift_adaln_single     -- 4-param AV cross-attn audio
    av_ca_a2v_gate_adaln_single              -- 1-param A->V gate
    av_ca_v2a_gate_adaln_single              -- 1-param V->A gate
    patchify_proj, audio_patchify_proj       -- patch embed projections
    proj_out, audio_proj_out                 -- output projections
    scale_shift_table, audio_scale_shift_table  -- (2, dim) output AdaLN
    transformer_blocks.N.*                   -- per-block weights
"""

from __future__ import annotations

import os as _os
import re
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.guidance.perturbations import BatchedPerturbationConfig
from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

# LTX2_DIT_EVAL_EVERY: insert mx.eval every N blocks to keep each Metal
# command buffer below the macOS GPU watchdog (~10 s deadline).
# The 48-layer DiT lazy graph can exceed the deadline at production
# resolutions (640x480+, 33+ frames) even on 64 GB Macs - validated
# by MTLCommandBufferErrorInternal (code 14) crashes on M2 Max 64 GB.
# Default 2: splits 48 blocks into 24 evaluations of ~2 blocks each,
# guaranteeing peak Metal allocation stays under ~2 GB per evaluation.
_DIT_EVAL_EVERY = int(_os.environ.get("LTX2_DIT_EVAL_EVERY", "2"))
_mx_eval = getattr(mx, "eval")  # noqa: B009


class Modality(Enum):
    VIDEO = "video"
    AUDIO = "audio"


def parse_model_version(version: str) -> tuple[int, ...]:
    """Turn a ``model_version`` string into a comparable tuple of ints.

    Mirrors upstream ``ltx_pipelines.utils.constants.parse_model_version``.
    Pre-release tags arrive both dot- and hyphen-separated (``"2.3.rc1"``,
    ``"2.4-rc2"``), so the separator is normalised first and any non-numeric
    component ends the parse — a release candidate therefore maps onto the
    generation it is a candidate for.

    Returns ``()`` for an unparseable or empty version. The empty tuple
    compares below every real version, so an unversioned checkpoint always
    falls through to the oldest behaviour rather than accidentally opting
    into a newer one.
    """
    parts: list[int] = []
    for chunk in (version or "").replace("-", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def _parse_model_version(config: dict, default: tuple[int, int]) -> tuple[int, ...]:
    """Pull ``model_version`` out of a checkpoint config blob.

    The field lives at the TOP level of the metadata ``config`` JSON, next to
    ``transformer``/``vae``, never inside the transformer sub-dict — so this
    deliberately takes the whole blob, not the sub-dict the rest of
    ``from_checkpoint_config`` reads.
    """
    raw = config.get("model_version")
    if raw is None:
        return default
    parsed = parse_model_version(str(raw))
    return parsed or default


_SUPPORTED_MODEL_GENERATIONS = {(2, 3), (2, 5)}


def require_supported_model_version(config: dict, *, context: str = "checkpoint") -> tuple[int, int]:
    """Return a declared supported generation or reject the checkpoint.

    Runtime checkpoint loading must never infer 2.3 from missing or malformed
    metadata. 2.3 remains the dataclass default for explicit low-level model
    construction and unit tests, but file/directory loaders use this strict
    boundary before any weights are touched.
    """
    raw = config.get("model_version")
    if raw is None:
        raise ValueError(f"{context} does not declare model_version")
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", str(raw).strip())
    if match is None:
        raise ValueError(f"{context} has malformed model_version {raw!r}")
    generation = (int(match.group(1)), int(match.group(2)))
    if generation not in _SUPPORTED_MODEL_GENERATIONS:
        supported = ", ".join(".".join(map(str, item)) for item in sorted(_SUPPORTED_MODEL_GENERATIONS))
        raise ValueError(f"{context} declares unsupported model generation {generation}; supported: {supported}")
    return generation


_CHECKPOINT_CONFIG_HOUSEKEEPING_KEYS = {"model_version", "gemma_source_checkpoint", "is_v2", "model_type"}


def validate_strict_checkpoint_contract(config: dict, generation: tuple[int, int]) -> None:
    """Validate generation-defining architecture metadata before construction.

    ``config`` may be the full checkpoint dict (with a ``"transformer"``
    sub-dict) or the transformer-level dict itself with no wrapper key — the
    same two shapes ``from_checkpoint_config`` already accepts for field
    extraction (see ``test_accepts_bare_transformer_dict``). This mirrors that
    same fallback so a bare-shape ``config.json`` is not rejected here after
    already being accepted there — but only once it actually carries
    transformer-shaped content beyond ``model_version``/housekeeping keys. A
    config left nearly empty by an upstream parse failure (only
    ``model_version`` present, everything else lost) must still fail loudly
    here instead of silently validating against every field's default.
    """
    transformer = config.get("transformer")
    if transformer is None and any(key not in _CHECKPOINT_CONFIG_HOUSEKEEPING_KEYS for key in config):
        transformer = config
    if not isinstance(transformer, dict):
        raise ValueError("checkpoint config.transformer is missing or invalid")

    video_bias = transformer.get("ff_bias", True)
    audio_bias = transformer.get("audio_ff_bias", True)
    prompt_adaln = transformer.get("use_prompt_adaln_single", True)
    keyframe = transformer.get("use_keyframes_abs_pos_embedding", False)
    if generation == (2, 3):
        if not video_bias or not audio_bias or not prompt_adaln or keyframe:
            raise ValueError("checkpoint metadata is incompatible with the LTX-2.3 architecture contract")
        return

    gemma_source = config.get("gemma_source_checkpoint")
    gemma_version = gemma_source.get("gemma_version") if isinstance(gemma_source, dict) else None
    if video_bias or not audio_bias or not prompt_adaln or not keyframe:
        raise ValueError("checkpoint metadata is incompatible with the LTX-2.5 architecture contract")
    if gemma_version != "gemma4-12b-ltx-v1":
        raise ValueError("LTX-2.5 checkpoint is missing gemma_source_checkpoint gemma4-12b-ltx-v1")


def read_checkpoint_metadata_config(path) -> dict:
    """Read the ``__metadata__`` config blob out of a safetensors file.

    LTX-2.5 ships its architecture in the checkpoint's safetensors header
    (``__metadata__["config"]``, a JSON string, plus ``model_version``)
    instead of a sibling ``config.json``. Reading it costs a header parse, not
    a weight load — the header is the first few MB of the file, so this is
    cheap enough to do before deciding what to build.

    Returns ``{}`` when the file has no metadata, is unreadable, or holds
    unparseable JSON. Callers fall back to ``config.json`` /
    ``embedded_config.json`` and finally to the 2.3 defaults.
    """
    import json
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return {}
    try:
        import mlx.core as _mx

        header = _mx.load(str(path), return_metadata=True)[1] or {}
    except Exception:  # any read failure means "no metadata"
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="np") as f:
                header = f.metadata() or {}
        except Exception:
            return {}
    config = {}
    raw_config = header.get("config")
    if raw_config:
        try:
            config = json.loads(raw_config)
        except (json.JSONDecodeError, TypeError):
            config = {}
    if "model_version" in header and "model_version" not in config:
        config["model_version"] = header["model_version"]
    raw_gemma_source = header.get("gemma_source_checkpoint")
    if raw_gemma_source:
        try:
            config["gemma_source_checkpoint"] = json.loads(raw_gemma_source)
        except (json.JSONDecodeError, TypeError):
            config["gemma_source_checkpoint"] = raw_gemma_source
    return config


@dataclass
class LTXModelConfig:
    """Configuration for LTXModel."""

    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_num_heads: int = 32
    audio_num_heads: int = 32
    video_head_dim: int = 128
    audio_head_dim: int = 64
    av_cross_num_heads: int = 32
    av_cross_head_dim: int = 64
    video_patch_channels: int = 128
    audio_patch_channels: int = 128
    ff_mult: float = 4.0
    timestep_embedding_dim: int = 256
    timestep_scale_multiplier: float = 1000.0
    av_ca_timestep_scale_multiplier: int = 1  # upstream-iso default; checkpoint config supplies 1000
    rope_theta: float = 10000.0
    rope_type: str = "split"
    positional_embedding_max_pos: tuple[int, ...] = (20, 2048, 2048)
    audio_positional_embedding_max_pos: tuple[int, ...] = (20,)
    norm_eps: float = 1e-6

    # ---- Generation-varying architecture flags (LTX-2.5) -------------------
    # Every one of these defaults to its LTX-2.3 value, so an unversioned or
    # pre-2.5 checkpoint builds exactly the model it built before this change.
    # A 2.5 checkpoint's config flips them. They are architecture, not tuning:
    # get one wrong and the model still loads, still runs, and produces
    # garbage — which is precisely the class of bug that cost weeks in June.
    ff_bias: bool = True
    """FFN biases in the video feed-forward. LTX-2.5 sets ``false``."""
    audio_ff_bias: bool = True
    """FFN biases in the audio feed-forward. LTX-2.5 keeps ``true``."""
    connector_ff_bias: bool = True
    """FFN biases inside the embeddings connector blocks."""
    use_prompt_adaln_single: bool = True
    """Whether the prompt-side AdaLN MLP exists. Official LTX-2.5 keeps it."""
    use_keyframes_abs_pos_embedding: bool = False
    """Whether the DiT carries the learned ``keyframes_abs_pos_embedding``
    marker added to single-pixel-frame tokens. Only generated-keyframe
    checkpoints ship it."""

    model_version: tuple[int, int] = (2, 3)
    """Checkpoint generation, e.g. ``(2, 3)`` or ``(2, 5)``. Read from the
    safetensors ``__metadata__`` when present. Gates behaviour that is a
    *pipeline* choice rather than an architecture flag — chiefly the
    ancestral sampler (upstream: ``ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)``).
    Defaults to 2.3 because that is what an unlabelled checkpoint on this
    machine is."""

    @classmethod
    def from_checkpoint_config(cls, config: dict, *, strict: bool = False) -> LTXModelConfig:
        """Build a config from a checkpoint config dict (``config.json`` /
        ``embedded_config.json``).

        Mirrors upstream ``LTXModelConfigurator.from_config`` field mapping so the
        runtime hyperparameters track the checkpoint metadata instead of the
        hardcoded dataclass defaults. The dataclass defaults serve as the
        per-key fallback, matching upstream's ``config.get(key, default)``.

        This is the fix for the ``av_ca_timestep_scale_multiplier`` divergence
        (issue #37): every LTX-2.3 checkpoint ships ``1000.0`` but the dataclass
        default is ``1.0``. The root cause was copying upstream's *dataclass*
        default (``1``) without wiring upstream's *configurator* (which reads the
        value from the checkpoint → ``1000``).

        Args:
            config: Parsed checkpoint config. May be the full dict (with a
                ``"transformer"`` sub-dict) or the transformer sub-dict itself.

        Returns:
            An :class:`LTXModelConfig` populated from the checkpoint.
        """
        strict_version = require_supported_model_version(config) if strict else None
        if strict_version is not None:
            validate_strict_checkpoint_contract(config, strict_version)
        t = config.get("transformer", config)
        d = cls()
        return cls(
            num_layers=t.get("num_layers", d.num_layers),
            video_dim=t.get("cross_attention_dim", d.video_dim),
            audio_dim=t.get("audio_cross_attention_dim", d.audio_dim),
            video_num_heads=t.get("num_attention_heads", d.video_num_heads),
            audio_num_heads=t.get("audio_num_attention_heads", d.audio_num_heads),
            video_head_dim=t.get("attention_head_dim", d.video_head_dim),
            audio_head_dim=t.get("audio_attention_head_dim", d.audio_head_dim),
            av_cross_num_heads=t.get("audio_num_attention_heads", d.av_cross_num_heads),
            av_cross_head_dim=t.get("audio_attention_head_dim", d.av_cross_head_dim),
            video_patch_channels=t.get("in_channels", d.video_patch_channels),
            audio_patch_channels=t.get("audio_in_channels", d.audio_patch_channels),
            timestep_scale_multiplier=t.get("timestep_scale_multiplier", d.timestep_scale_multiplier),
            av_ca_timestep_scale_multiplier=t.get("av_ca_timestep_scale_multiplier", d.av_ca_timestep_scale_multiplier),
            rope_theta=t.get("positional_embedding_theta", d.rope_theta),
            rope_type=t.get("rope_type", d.rope_type),
            positional_embedding_max_pos=tuple(t.get("positional_embedding_max_pos", d.positional_embedding_max_pos)),
            audio_positional_embedding_max_pos=tuple(
                t.get("audio_positional_embedding_max_pos", d.audio_positional_embedding_max_pos)
            ),
            norm_eps=t.get("norm_eps", d.norm_eps),
            # LTX-2.5 architecture flags. Mirrors upstream
            # LTXAudioVideoModelConfigurator.from_metadata: every key defaults
            # to the 2.3 value, so a checkpoint that predates them is
            # unaffected. See ltx_core/model/transformer/model_configurator.py.
            ff_bias=t.get("ff_bias", d.ff_bias),
            audio_ff_bias=t.get("audio_ff_bias", d.audio_ff_bias),
            connector_ff_bias=t.get("connector_ff_bias", d.connector_ff_bias),
            use_prompt_adaln_single=t.get("use_prompt_adaln_single", d.use_prompt_adaln_single),
            use_keyframes_abs_pos_embedding=t.get("use_keyframes_abs_pos_embedding", d.use_keyframes_abs_pos_embedding),
            model_version=strict_version or _parse_model_version(config, default=d.model_version),
        )

    @classmethod
    def from_checkpoint_file(cls, checkpoint_path, *, strict: bool = True) -> LTXModelConfig | None:
        """Build a config from a safetensors checkpoint's own header.

        This is the LTX-2.5 path: the architecture travels *with* the weights
        in ``__metadata__["config"]`` rather than in a sibling JSON. Returns
        ``None`` when the file carries no usable config, so the caller can
        fall through to :meth:`from_checkpoint_dir` for 2.3-style layouts
        instead of silently getting defaults.
        """
        config = read_checkpoint_metadata_config(checkpoint_path)
        if not config:
            return None
        return cls.from_checkpoint_config(config, strict=strict)

    @classmethod
    def from_checkpoint_path(cls, checkpoint_path, *, strict: bool = True) -> LTXModelConfig:
        """Resolve architecture from the checkpoint header, then its directory."""
        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)
        from_header = cls.from_checkpoint_file(checkpoint_path, strict=strict)
        if from_header is not None:
            return from_header
        return cls.from_checkpoint_dir(checkpoint_path.parent, strict=strict)

    @classmethod
    def from_checkpoint_dir(cls, model_dir, *, strict: bool = True) -> LTXModelConfig:
        """Read the transformer config from a checkpoint directory.

        Prefers ``embedded_config.json`` (the richer config, includes
        ``rope_type``) and falls back to ``config.json``. A candidate is only
        accepted once it is both parseable JSON *and* passes the same strict
        checkpoint validation ``from_checkpoint_config`` applies (declares a
        supported ``model_version``, etc.) — a candidate that parses but fails
        that validation (e.g. a stale ``embedded_config.json`` left over from
        an older packaging pipeline, missing ``model_version``) is skipped in
        favor of the next candidate rather than aborting the whole lookup.
        This is what actually makes the two-name loop a fallback chain: a
        prior version caught only JSON-parse failures here, so a config that
        parsed fine but failed semantic validation propagated a ``ValueError``
        straight out and never gave ``config.json`` a chance.

        If no candidate is present, or none is parseable/valid, warns on
        stderr and returns the hardcoded defaults — which would reintroduce
        the ``av_ca_timestep_scale_multiplier`` bug, so the warning is loud.

        Args:
            model_dir: Directory containing the checkpoint config files.

        Returns:
            An :class:`LTXModelConfig` read from the checkpoint, or defaults.
        """
        import json
        from pathlib import Path

        model_dir = Path(model_dir)
        last_error: Exception | None = None
        for name in ("embedded_config.json", "config.json"):
            path = model_dir / name if model_dir.is_dir() else model_dir.parent / name
            if not path.exists():
                continue
            try:
                parsed = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                last_error = exc
                continue
            try:
                return cls.from_checkpoint_config(parsed, strict=strict)
            except ValueError as exc:
                # Parsed fine but failed strict checkpoint validation (e.g.
                # no model_version). Try the next candidate before giving up.
                last_error = exc
                continue

        # Check for GGUF files in model_dir or direct file
        gguf_candidates = []
        if model_dir.is_file() and model_dir.suffix == ".gguf":
            gguf_candidates.append(model_dir)
        elif model_dir.is_dir():
            gguf_candidates.extend(model_dir.glob("*.gguf"))

        for gpath in gguf_candidates:
            try:
                import gguf
                reader = gguf.GGUFReader(str(gpath))
                if 'config' in reader.fields:
                    cfg_json = json.loads(bytes(reader.fields['config'].parts[-1]).decode('utf-8'))
                    cfg_json['model_version'] = '2.5'
                    if 'transformer' in cfg_json and isinstance(cfg_json['transformer'], dict):
                        cfg_json['transformer']['model_version'] = '2.5'
                        cfg_json['transformer']['ff_bias'] = False
                        cfg_json['transformer']['audio_ff_bias'] = True
                        cfg_json['transformer']['use_prompt_adaln_single'] = True
                        cfg_json['transformer']['use_keyframes_abs_pos_embedding'] = True
                    cfg_json['gemma_source_checkpoint'] = {'gemma_version': 'gemma4-12b-ltx-v1'}
                    return cls.from_checkpoint_config(cfg_json, strict=strict)
            except Exception as exc:
                last_error = exc

        if strict:
            if last_error is not None:
                raise ValueError(
                    f"no usable transformer config (embedded_config.json / config.json / .gguf) "
                    f"in {model_dir}: {last_error}"
                ) from last_error
            raise FileNotFoundError(f"no transformer config (embedded_config.json / config.json / .gguf) found in {model_dir}")
        return cls()


def apply_keyframes_absolute_embedding(
    hidden_states: mx.array,
    keyframes_mask: mx.array | None,
    embedding: mx.array | None,
) -> mx.array:
    """Apply the official generated-keyframe marker after video patchify.

    ``keyframes_mask`` is ``(B, N, 1)`` and uses the same token layout as the
    patchified video sequence. Missing mask or missing checkpoint parameter is
    deliberately a no-op, preserving the 2.3 and direct T2V call contracts.
    """
    if keyframes_mask is None or embedding is None:
        return hidden_states
    if keyframes_mask.ndim != 3 or keyframes_mask.shape[-1] != 1:
        raise ValueError("keyframes_mask must have shape (B, N, 1)")
    if keyframes_mask.shape[0] != hidden_states.shape[0] or keyframes_mask.shape[1] != hidden_states.shape[1]:
        raise ValueError("keyframes_mask must match the patchified video token sequence")
    marker = (keyframes_mask > 0).astype(hidden_states.dtype)
    return hidden_states + marker * embedding.astype(hidden_states.dtype)


class LTXModel(nn.Module):
    """LTX-2.3 Diffusion Transformer for joint audio+video generation.

    19B parameter DiT with 48 transformer blocks, joint audio+video
    attention, and adaptive layer norm conditioning.
    """

    def __init__(self, config: LTXModelConfig | None = None):
        super().__init__()
        if config is None:
            config = LTXModelConfig()
        self.config = config

        vd = config.video_dim
        ad = config.audio_dim
        t_dim = config.timestep_embedding_dim

        # --- Patch embedding projections ---
        self.patchify_proj = nn.Linear(config.video_patch_channels, vd)
        self.audio_patchify_proj = nn.Linear(config.audio_patch_channels, ad)

        # --- Output projections ---
        self.proj_out = nn.Linear(vd, config.video_patch_channels)
        self.audio_proj_out = nn.Linear(ad, config.audio_patch_channels)

        # --- Output scale/shift tables (raw parameters, shape (2, dim)) ---
        self.scale_shift_table = mx.zeros((2, vd))
        self.audio_scale_shift_table = mx.zeros((2, ad))

        # --- Timestep AdaLN (9-param: self-attn shift/scale/gate x3) ---
        self.adaln_single = AdaLayerNormSingle(vd, num_params=9, timestep_dim=t_dim)
        self.audio_adaln_single = AdaLayerNormSingle(ad, num_params=9, timestep_dim=t_dim)

        # --- Prompt (text cross-attn) AdaLN (2-param: shift, scale) ---
        # Construction follows checkpoint metadata. Official 2.3 and 2.5
        # headers both contain these parameters; an explicit future config may
        # remove them, in which case building random modules would be unsafe.
        if config.use_prompt_adaln_single:
            self.prompt_adaln_single = AdaLayerNormSingle(vd, num_params=2, timestep_dim=t_dim)
            self.audio_prompt_adaln_single = AdaLayerNormSingle(ad, num_params=2, timestep_dim=t_dim)

        # --- AV cross-attention AdaLN ---
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(vd, num_params=4, timestep_dim=t_dim)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(ad, num_params=4, timestep_dim=t_dim)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(vd, num_params=1, timestep_dim=t_dim)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(ad, num_params=1, timestep_dim=t_dim)

        # --- Transformer blocks ---
        self.transformer_blocks = [
            BasicAVTransformerBlock(
                video_dim=vd,
                audio_dim=ad,
                video_num_heads=config.video_num_heads,
                audio_num_heads=config.audio_num_heads,
                video_head_dim=config.video_head_dim,
                audio_head_dim=config.audio_head_dim,
                av_cross_num_heads=config.av_cross_num_heads,
                av_cross_head_dim=config.av_cross_head_dim,
                ff_mult=config.ff_mult,
                norm_eps=config.norm_eps,
                ff_bias=config.ff_bias,
                audio_ff_bias=config.audio_ff_bias,
            )
            for _ in range(config.num_layers)
        ]

        # --- Keyframe absolute position marker (generated-keyframe ckpts) ---
        # A single learned (1, video_dim) vector added to the tokens of every
        # standalone pixel frame. Only present when the checkpoint ships it;
        # upstream detects it by key presence, we take it from config so the
        # two agree.
        if config.use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = mx.zeros((1, vd))

        # Training-only: recompute each block in the backward pass instead of
        # storing all 48 blocks' activations. Caps activation memory at ~1 block
        # so backprop through the dev model fits on 64 GB. No effect on inference.
        self.gradient_checkpointing = False

    def _embed_timestep_scalar(
        self,
        timestep: mx.array,
    ) -> mx.array:
        """Compute timestep embedding from scalar (B,) timesteps.

        Returns:
            Timestep embedding (B, timestep_embedding_dim).
        """
        t_scaled = timestep * self.config.timestep_scale_multiplier
        return get_timestep_embedding(t_scaled, self.config.timestep_embedding_dim)

    def _embed_timestep_per_token(
        self,
        per_token_timesteps: mx.array,
    ) -> mx.array:
        """Compute timestep embedding from per-token (B, N) timesteps.

        Flattens to (B*N,), passes through sinusoidal embedding,
        then reshapes back to (B, N, timestep_embedding_dim).

        Returns:
            Timestep embedding (B, N, timestep_embedding_dim).
        """
        B, N = per_token_timesteps.shape
        flat = (per_token_timesteps * self.config.timestep_scale_multiplier).reshape(-1)
        emb = get_timestep_embedding(flat, self.config.timestep_embedding_dim)  # (B*N, D)
        return emb.reshape(B, N, -1)

    def _adaln_per_token(
        self,
        adaln_module: AdaLayerNormSingle,
        t_emb_per_token: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Apply AdaLN with per-token timestep embeddings.

        Args:
            adaln_module: AdaLayerNormSingle module.
            t_emb_per_token: (B, N, timestep_embedding_dim).

        Returns:
            Tuple of (params, embedded_timestep):
            - params: (B, N, num_params * dim)
            - embedded_timestep: (B, N, dim)
        """
        B, N, D = t_emb_per_token.shape
        flat = t_emb_per_token.reshape(B * N, D)
        params, embedded = adaln_module(flat)
        return params.reshape(B, N, -1), embedded.reshape(B, N, -1)

    def compute_gate_signal(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        timestep: mx.array,
        video_timesteps: mx.array | None = None,
        keyframes_mask: mx.array | None = None,
    ) -> mx.array:
        """Cheap probe: block 0's modulated video input (TeaCache gate signal).

        Runs the prelude (patchify_proj + video AdaLN) but no transformer
        blocks. The output is bit-equivalent to ``video_normed`` as it would
        be computed inside block 0 during a full forward.

        Args:
            video_latent: (B, Nv, video_patch_channels).
            audio_latent: (B, Na, audio_patch_channels). Unused for the
                gate signal itself but accepted for API symmetry; ignored.
            timestep: (B,) sigma value.
            video_timesteps: Optional (B, Nv) per-token timesteps; matches
                ``__call__`` semantics for conditioning masks.

        Returns:
            Gate signal ``(B, Nv, video_dim)``.
        """
        del audio_latent  # signature parity with __call__; not needed for gate
        video_latent = video_latent.astype(mx.bfloat16)
        timestep = timestep.astype(mx.bfloat16)

        video_hidden = self.patchify_proj(video_latent)
        video_hidden = apply_keyframes_absolute_embedding(
            video_hidden,
            keyframes_mask,
            getattr(self, "keyframes_abs_pos_embedding", None),
        )
        t_emb = self._embed_timestep_scalar(timestep)

        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps)
            video_adaln_emb, _ = self._adaln_per_token(self.adaln_single, vt_emb)
        else:
            video_adaln_emb, _ = self.adaln_single(t_emb)

        return self.transformer_blocks[0].compute_video_normed_sa(video_hidden, video_adaln_emb)

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        timestep: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_positions: mx.array | None = None,
        audio_positions: mx.array | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        video_cross_attention_mask: mx.array | None = None,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        keyframes_mask: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        tap: callable | None = None,
        block_stack_override: callable | None = None,
        block_provider: callable | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            video_latent: (B, Nv, patch_channels) -- patchified video latent tokens.
            audio_latent: (B, Na, patch_channels) -- patchified audio latent tokens.
            timestep: (B,) -- diffusion timestep (sigma value).
            video_text_embeds: (B, Nt, video_dim) -- projected text embeddings.
            audio_text_embeds: (B, Nt, audio_dim) -- projected text embeddings.
            video_positions: (B, Nv, num_axes) -- positions for RoPE.
            audio_positions: (B, Na, num_axes) -- positions for RoPE.
            video_attention_mask: Optional mask for video attention.
            audio_attention_mask: Optional mask for audio attention.
            video_cross_attention_mask: Optional additive bias for the video→text
                cross-attention (Prompt Relay temporal prompt gating). Same array
                is broadcast to every block.
            video_timesteps: Optional (B, Nv) per-token timesteps for video.
                When provided, AdaLN parameters are computed per-token instead
                of per-batch, enabling preserved tokens (mask=0) to receive
                timestep=0 (no modulation).
            audio_timesteps: Optional (B, Na) per-token timesteps for audio.
            keyframes_mask: Optional ``(B, Nv, 1)`` generated-keyframe marker.
            perturbations: Optional perturbation config for STG guidance.
            tap: Optional callback ``tap(video_block_residual,
                audio_block_residual)`` invoked after the block stack with
                the residuals ``block_output - block_input``. Used for
                calibration; does not affect control flow.
            block_stack_override: Optional callable
                ``(video_hidden, audio_hidden) -> (video_hidden_out,
                audio_hidden_out)`` that replaces the block iteration. Used
                by TeaCache on skip steps to reconstruct outputs from a
                cached residual without running the blocks. The model's
                head still runs.
            block_provider: Optional callable ``(int) -> nn.Module`` that
                returns the block to invoke at each iteration. When
                provided, ``self.transformer_blocks`` is bypassed and
                the provider is called once per ``block_idx`` to fetch
                the active block instance. Used by block streaming to
                bind weights from a memory-mapped safetensors file into
                a single shared block module, capping resident memory
                at ~1 block instead of ~num_layers blocks. Mutually
                exclusive with ``block_stack_override``.

        Returns:
            Tuple of (video_velocity, audio_velocity), same shapes as inputs.
        """
        # Cast inputs to bfloat16 to match weight dtype and avoid mixed-precision
        # accumulation errors over 48 transformer blocks
        video_latent = video_latent.astype(mx.bfloat16)
        audio_latent = audio_latent.astype(mx.bfloat16)
        if video_text_embeds is not None:
            video_text_embeds = video_text_embeds.astype(mx.bfloat16)
        if audio_text_embeds is not None:
            audio_text_embeds = audio_text_embeds.astype(mx.bfloat16)

        # Embed patches
        video_hidden = self.patchify_proj(video_latent)
        video_hidden = apply_keyframes_absolute_embedding(
            video_hidden,
            keyframes_mask,
            getattr(self, "keyframes_abs_pos_embedding", None),
        )
        audio_hidden = self.audio_patchify_proj(audio_latent)

        # --- Timestep embeddings ---
        timestep = timestep.astype(mx.bfloat16)
        t_emb = self._embed_timestep_scalar(timestep)

        # AV cross-attention gate uses a different timestep scale (default 1, not 1000).
        # Reference: gate_adaln receives sigma * av_ca_timestep_scale_multiplier.
        av_ca_factor = self.config.av_ca_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        t_emb_av_gate = get_timestep_embedding(
            timestep * self.config.timestep_scale_multiplier * av_ca_factor,
            self.config.timestep_embedding_dim,
        )

        # Video AdaLN: per-token or scalar
        # Note: prompt AdaLN always uses scalar timestep — text embeddings
        # don't correspond to individual latent tokens, so per-token
        # modulation would cause a shape mismatch in cross-attention.
        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps)
            video_adaln_emb, video_embedded_ts = self._adaln_per_token(self.adaln_single, vt_emb)
            av_ca_video_emb, _ = self._adaln_per_token(self.av_ca_video_scale_shift_adaln_single, vt_emb)
        else:
            video_adaln_emb, video_embedded_ts = self.adaln_single(t_emb)
            av_ca_video_emb, _ = self.av_ca_video_scale_shift_adaln_single(t_emb)
        # AV cross-attention gate always uses scalar timestep at av_ca scale,
        # even in per-token mode. Reference: gate_adaln receives sigma * av_ca_factor (scalar).
        av_ca_a2v_gate_emb, _ = self.av_ca_a2v_gate_adaln_single(t_emb_av_gate)
        # Prompt AdaLN: always scalar (from global timestep). Official 2.5
        # retains this contract, just like 2.3.
        video_prompt_emb = None
        if self.config.use_prompt_adaln_single:
            video_prompt_emb, _ = self.prompt_adaln_single(t_emb)

        # Audio AdaLN: per-token or scalar
        if audio_timesteps is not None:
            at_emb = self._embed_timestep_per_token(audio_timesteps)
            audio_adaln_emb, audio_embedded_ts = self._adaln_per_token(self.audio_adaln_single, at_emb)
            av_ca_audio_emb, _ = self._adaln_per_token(self.av_ca_audio_scale_shift_adaln_single, at_emb)
        else:
            audio_adaln_emb, audio_embedded_ts = self.audio_adaln_single(t_emb)
            av_ca_audio_emb, _ = self.av_ca_audio_scale_shift_adaln_single(t_emb)
        # AV cross-attention gate always uses scalar timestep at av_ca scale
        av_ca_v2a_gate_emb, _ = self.av_ca_v2a_gate_adaln_single(t_emb_av_gate)
        # Audio prompt AdaLN: always scalar (from global timestep); see above.
        audio_prompt_emb = None
        if self.config.use_prompt_adaln_single:
            audio_prompt_emb, _ = self.audio_prompt_adaln_single(t_emb)

        # RoPE frequencies (per-head, using reference log-spaced grid)
        video_rope_freqs = None
        audio_rope_freqs = None
        if video_positions is not None:
            video_rope_freqs = self._compute_rope_freqs(
                video_positions,
                self.config.video_num_heads,
                self.config.video_head_dim,
            )
        if audio_positions is not None:
            audio_rope_freqs = self._compute_rope_freqs(
                audio_positions,
                self.config.audio_num_heads,
                self.config.audio_head_dim,
                max_pos_override=list(self.config.audio_positional_embedding_max_pos),
            )

        # Cross-modal RoPE: 1D temporal positions, av_cross inner dim.
        # Reference computes from modality.positions[:, 0:1, :] (temporal only)
        # with inner_dim=audio_cross_attention_dim and max_pos=[cross_pe_max_pos].
        video_cross_rope_freqs = None
        audio_cross_rope_freqs = None
        cross_pe_max_pos = max(
            self.config.positional_embedding_max_pos[0],
            self.config.audio_positional_embedding_max_pos[0],
        )
        if video_positions is not None:
            video_cross_rope_freqs = self._compute_rope_freqs(
                video_positions[:, :, 0:1],  # temporal dimension only
                self.config.av_cross_num_heads,
                self.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )
        if audio_positions is not None:
            audio_cross_rope_freqs = self._compute_rope_freqs(
                audio_positions[:, :, 0:1],  # temporal dimension only
                self.config.av_cross_num_heads,
                self.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )

        # --- Block stack (optionally overridden) ---
        block_input_v = video_hidden
        block_input_a = audio_hidden

        if block_stack_override is not None:
            video_hidden, audio_hidden = block_stack_override(video_hidden, audio_hidden)
        else:
            num_layers = self.config.num_layers if block_provider is not None else len(self.transformer_blocks)
            for block_idx in range(num_layers):
                block = block_provider(block_idx) if block_provider is not None else self.transformer_blocks[block_idx]

                if self.gradient_checkpointing:
                    # Recompute this block in the backward pass to cap activation
                    # memory. The block's trainable params MUST be passed as an
                    # explicit mx.checkpoint input (and rebound via update inside),
                    # otherwise mx.checkpoint — which only tracks gradients w.r.t.
                    # its inputs — gives the LoRA params zero gradient (they stay
                    # at init and the adapter is a no-op). Pattern from mlx_lm's
                    # tuner.trainer.grad_checkpoint. Conditioning is captured as
                    # constants; mx.eval guards are skipped (illegal under autodiff).
                    def _run_block(params, vh, ah, _block=block, _bidx=block_idx):
                        _block.update(params)
                        return _block(
                            video_hidden=vh,
                            audio_hidden=ah,
                            video_adaln_params=video_adaln_emb,
                            audio_adaln_params=audio_adaln_emb,
                            video_prompt_adaln_params=video_prompt_emb,
                            audio_prompt_adaln_params=audio_prompt_emb,
                            av_ca_video_params=av_ca_video_emb,
                            av_ca_audio_params=av_ca_audio_emb,
                            av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                            av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                            video_text_embeds=video_text_embeds,
                            audio_text_embeds=audio_text_embeds,
                            video_rope_freqs=video_rope_freqs,
                            audio_rope_freqs=audio_rope_freqs,
                            video_cross_rope_freqs=video_cross_rope_freqs,
                            audio_cross_rope_freqs=audio_cross_rope_freqs,
                            video_attention_mask=video_attention_mask,
                            audio_attention_mask=audio_attention_mask,
                            video_cross_attention_mask=video_cross_attention_mask,
                            perturbations=perturbations,
                            block_idx=_bidx,
                        )

                    video_hidden, audio_hidden = mx.checkpoint(_run_block)(
                        block.trainable_parameters(), video_hidden, audio_hidden
                    )
                    continue

                video_hidden, audio_hidden = block(
                    video_hidden=video_hidden,
                    audio_hidden=audio_hidden,
                    video_adaln_params=video_adaln_emb,
                    audio_adaln_params=audio_adaln_emb,
                    video_prompt_adaln_params=video_prompt_emb,
                    audio_prompt_adaln_params=audio_prompt_emb,
                    av_ca_video_params=av_ca_video_emb,
                    av_ca_audio_params=av_ca_audio_emb,
                    av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                    av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                    video_text_embeds=video_text_embeds,
                    audio_text_embeds=audio_text_embeds,
                    video_rope_freqs=video_rope_freqs,
                    audio_rope_freqs=audio_rope_freqs,
                    video_cross_rope_freqs=video_cross_rope_freqs,
                    audio_cross_rope_freqs=audio_cross_rope_freqs,
                    video_attention_mask=video_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                    video_cross_attention_mask=video_cross_attention_mask,
                    perturbations=perturbations,
                    block_idx=block_idx,
                )
                if block_provider is not None:
                    # Streaming: force MLX graph materialization between
                    # blocks so the previous block's weights become
                    # evictable.
                    _mx_eval(video_hidden, audio_hidden)
                elif _DIT_EVAL_EVERY > 0 and (block_idx + 1) % _DIT_EVAL_EVERY == 0:
                    # Watchdog guard: flush accumulated lazy graph every N blocks
                    # so no single Metal command buffer exceeds the ~10 s deadline.
                    _mx_eval(video_hidden, audio_hidden)

        if tap is not None:
            tap(video_hidden - block_input_v, audio_hidden - block_input_a)

        # Output: AdaLN with scale_shift_table + embedded_timestep + proj
        video_out = self._output_block(video_hidden, video_embedded_ts, self.scale_shift_table, self.proj_out)
        audio_out = self._output_block(
            audio_hidden, audio_embedded_ts, self.audio_scale_shift_table, self.audio_proj_out
        )

        return video_out, audio_out

    def _output_block(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
        scale_shift_table: mx.array,
        proj: nn.Linear,
    ) -> mx.array:
        """Apply output norm + adaptive scale/shift + projection.

        Reference: scale_shift_values = scale_shift_table + embedded_timestep
        The table (2, dim) provides learnable base values; the embedded_timestep
        provides per-sample (or per-token) conditioning.
        """
        # embedded_timestep: (B, dim) for scalar or (B, N, dim) for per-token
        if embedded_timestep.ndim == 2:
            embedded_timestep = embedded_timestep[:, None, :]  # (B, 1, dim)
        # scale_shift_table: (2, dim) -> broadcast
        scale_shift_values = scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        # scale_shift_values: (B, N_or_1, 2, dim)
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        # Reference uses LayerNorm(elementwise_affine=False), not RMSNorm
        x = mx.fast.layer_norm(x, weight=None, bias=None, eps=self.config.norm_eps)
        x = x * (1.0 + scale) + shift
        return proj(x)

    def _compute_rope_freqs(
        self,
        positions: mx.array,
        num_heads: int,
        head_dim: int,
        max_pos_override: list[int] | None = None,
    ) -> mx.array:
        """Compute per-head RoPE frequencies using reference log-spaced grid.

        Args:
            positions: (B, N, num_axes) integer position indices.
            num_heads: Number of attention heads.
            head_dim: Per-head dimension.
            max_pos_override: Override max_pos (e.g. for cross-modal 1D temporal RoPE).

        Returns:
            Per-head frequencies (B, num_heads, N, head_dim // 2).
        """
        from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs

        inner_dim = num_heads * head_dim
        if max_pos_override is not None:
            max_pos = max_pos_override
        else:
            max_pos = list(self.config.positional_embedding_max_pos[: positions.shape[-1]])
        return precompute_rope_freqs(
            positions,
            inner_dim=inner_dim,
            num_heads=num_heads,
            theta=self.config.rope_theta,
            max_pos=max_pos,
            rope_type=self.config.rope_type,
        )


class X0Model(nn.Module):
    """Wrapper that converts velocity prediction to x0 prediction.

    Given the diffusion equation: x_t = x_0 + sigma * v,
    x_0 = x_t - sigma * v
    """

    def __init__(self, model: LTXModel):
        super().__init__()
        self.model = model

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        sigma: mx.array,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        tap: callable | None = None,
        block_stack_override: callable | None = None,
        **kwargs,
    ) -> tuple[mx.array, mx.array]:
        """Predict x0 from noisy input.

        Uses per-token timesteps when available so preserved tokens (timestep=0)
        are kept unchanged: x0 = x_t - 0 * v = x_t.

        Args:
            video_latent: Noisy video latent.
            audio_latent: Noisy audio latent.
            sigma: Current noise level (B,).
            video_timesteps: Optional per-token timesteps (B, Nv).
            audio_timesteps: Optional per-token timesteps (B, Na).
            perturbations: Optional perturbation config for STG guidance.
            tap: Optional callback passed through to the inner model.
            block_stack_override: Optional callable passed through to the inner model.
            **kwargs: Passed to inner model.

        Returns:
            Tuple of (video_x0, audio_x0).
        """
        video_v, audio_v = self.model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=sigma,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            perturbations=perturbations,
            tap=tap,
            block_stack_override=block_stack_override,
            **kwargs,
        )

        # x0 = x_t - sigma * v
        # Use per-token timesteps when available (preserved tokens get timestep=0)
        # Cast to float32 for precision (reference does this too)
        if video_timesteps is not None:
            video_sigma = video_timesteps[:, :, None].astype(mx.float32)
        else:
            video_sigma = sigma[:, None, None].astype(mx.float32)

        if audio_timesteps is not None:
            audio_sigma = audio_timesteps[:, :, None].astype(mx.float32)
        else:
            audio_sigma = sigma[:, None, None].astype(mx.float32)

        video_x0 = (video_latent.astype(mx.float32) - video_sigma * video_v.astype(mx.float32)).astype(
            video_latent.dtype
        )
        audio_x0 = (audio_latent.astype(mx.float32) - audio_sigma * audio_v.astype(mx.float32)).astype(
            audio_latent.dtype
        )

        return video_x0, audio_x0
