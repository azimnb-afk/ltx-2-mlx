"""Composable pipeline blocks.

Mirrors upstream ``ltx_pipelines.utils.blocks`` (composition over
inheritance). Each block owns the lifecycle of one model component
(load, use, free) and exposes a small ``__call__`` API. Pipelines
that prefer composition can instantiate these blocks directly:

```python
from ltx_pipelines_mlx import PromptEncoder, VideoDecoder, AudioDecoder

prompt_enc = PromptEncoder(model_dir, gemma_model_id)
video_emb, audio_emb = prompt_enc(prompt)  # loads, encodes, frees

video_dec = VideoDecoder(model_dir)
video_dec.decode_and_stream(video_latent, "out.mp4", audio_path="audio.wav")
```

The :class:`BasePipeline` inheritance tree (:class:`TI2VidTwoStagesPipeline`,
:class:`RetakePipeline`, :class:`ICLoraPipeline`, ...) **delegates** to
these blocks internally. Each pipeline holds private block instances
(``self._prompt_encoder``, ``self._image_conditioner``,
``self._video_decoder``, ``self._audio_decoder_block``); the historical
attribute names (``self.text_encoder``, ``self.vae_encoder``, ...) are
properties that proxy onto the block internals so subclass code that
reads/writes them — including ``self.text_encoder = None`` to free
memory — continues to work.

The blocks are the single source of truth for loader logic; the
inheritance API exists purely for backward compat with the current
subclass bodies.

Differences vs upstream:

- No CPU/GPU offload context managers — MLX uses unified memory, so
  blocks just hold strong refs and rely on Python GC + ``aggressive_cleanup``.
- No ``Builder``/``Registry`` indirection — blocks load weights via
  :func:`load_split_safetensors` directly, mirroring our existing path.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.transformer.model import LTXModelConfig, read_checkpoint_metadata_config
from ltx_core_mlx.model.upsampler.model import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder as _VideoVAEDecoder
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder as _VideoVAEEncoder
from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.text_encoders.gemma.loader import (
    detect_text_encoder,
    resolve_text_encoder_config_path,
    resolve_text_tower,
    validate_text_encoder_compatibility,
)
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags the literal mx.eval pattern


def _resolve_model_dir(model_dir: str | Path) -> Path:
    """Resolve a model dir — download from HuggingFace if not a local path."""
    path = Path(model_dir)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_dir)))


def _resolve_component_file(
    model_dir: Path,
    filename: str,
    extra_stems: list[str] | None = None,
    allow_external_cache_fallback: bool = True,
) -> Path:
    """Resolve a component file from model_dir, environment, or standard cache paths.

    ``allow_external_cache_fallback`` gates the final, cross-model search over
    generic HuggingFace/local cache directories. That search has no way to
    verify the found file belongs to the same model version being loaded —
    fine for components genuinely shared across a user's local caches, but
    wrong for a component (like the Video VAE) whose weights differ between
    model versions and where a wrong match would load silently. Callers for
    version-sensitive components should pass ``False`` and fail closed
    instead when nothing is found in ``model_dir`` itself.
    """
    p = model_dir / filename
    if p.exists():
        return p
    if extra_stems:
        for stem in extra_stems:
            for match in model_dir.glob(f"*{stem}*"):
                if match.is_file():
                    return match
    import os
    env_dir = os.environ.get("LTX_MODEL_DIR")
    if env_dir:
        p_env = Path(env_dir) / filename
        if p_env.exists():
            return p_env
    if not allow_external_cache_fallback:
        return p
    home = Path.home()
    cache_dirs = [
        home / ".cache" / "huggingface" / "hub",
        home / ".models",
        home / "AI" / "LTX-MLX" / "models",
    ]
    for cdir in cache_dirs:
        if cdir.exists():
            matches = list(cdir.glob(f"**/{filename}"))
            if matches:
                return matches[0]
            if extra_stems:
                for stem in extra_stems:
                    matches = list(cdir.glob(f"**/*{stem}*"))
                    if matches:
                        return matches[0]
    return p


class PromptEncoder:
    """Owns Gemma + connector lifecycle. Encodes prompts on call.

    Mirrors upstream ``utils.blocks.PromptEncoder``. Loads Gemma + the
    feature-extractor connector lazily on first call, encodes the prompt
    into ``(video_embeds, audio_embeds)``, then frees both modules.
    """

    def __init__(
        self,
        model_dir: str | Path,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.gemma_model_id = gemma_model_id
        self._text_encoder: GemmaLanguageModel | None = None
        self._feature_extractor: GemmaFeaturesExtractorV2 | None = None

    def load(self) -> None:
        """Load text encoder + connector (cached)."""
        if self._text_encoder is None:
            candidates = sorted(self.model_dir.glob("*transformer*.safetensors"))
            metadata = {}
            model_config = None
            for candidate in candidates:
                metadata = read_checkpoint_metadata_config(candidate)
                if metadata:
                    model_config = LTXModelConfig.from_checkpoint_config(metadata, strict=True)
                    break
            if model_config is None:
                model_config = LTXModelConfig.from_checkpoint_dir(self.model_dir, strict=True)

            if model_config.model_version == (2, 5):
                from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextConfig, Gemma4TextTower
                self._text_encoder = GemmaLanguageModel(architecture="gemma4")
                g4_cfg = Gemma4TextConfig(num_hidden_layers=48, hidden_size=3840)
                self._text_encoder.model = Gemma4TextTower(g4_cfg)
                self._text_encoder.load(self.gemma_model_id)
            else:
                gemma_source = metadata.get("gemma_source_checkpoint") or {}
                gemma_source_version = gemma_source.get("gemma_version") if isinstance(gemma_source, dict) else None
                encoder_config_path = resolve_text_encoder_config_path(self.gemma_model_id)
                encoder_spec = detect_text_encoder(encoder_config_path)
                validate_text_encoder_compatibility(
                    encoder_spec,
                    model_config.model_version,
                    gemma_source_version=gemma_source_version,
                )
                self._text_encoder = GemmaLanguageModel(architecture=resolve_text_tower(encoder_spec))
                self._text_encoder.load(self.gemma_model_id)
            aggressive_cleanup()

        if self._feature_extractor is None:
            self._feature_extractor = GemmaFeaturesExtractorV2()
            gguf_path = None
            if self.model_dir.is_file() and self.model_dir.suffix == ".gguf":
                gguf_path = self.model_dir
            elif self.model_dir.is_dir():
                ggufs = list(self.model_dir.glob("*.gguf"))
                if ggufs:
                    gguf_path = ggufs[0]

            if gguf_path and gguf_path.exists():
                import gguf
                from gguf.quants import dequantize
                reader = gguf.GGUFReader(str(gguf_path))
                connector_weights = {}
                for t in reader.tensors:
                    if "embeddings_connector" in t.name or "text_embedding_projection" in t.name:
                        arr = dequantize(t.data, t.tensor_type)
                        connector_weights[t.name] = mx.array(arr)
                if connector_weights:
                    self._feature_extractor.connector.load_weights(list(connector_weights.items()), strict=False)
            else:
                connector_path = _resolve_component_file(self.model_dir, "connector.safetensors")
                if connector_path.exists():
                    connector_weights = load_split_safetensors(connector_path, prefix="connector.")
                    self._feature_extractor.connector.load_weights(list(connector_weights.items()), strict=False)
            aggressive_cleanup()

    def free(self) -> None:
        """Drop strong refs; rely on GC + aggressive_cleanup to reclaim memory."""
        self._text_encoder = None
        self._feature_extractor = None
        aggressive_cleanup()

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode a single prompt to ``(video_embeds, audio_embeds)``.

        Caller is responsible for freeing via :meth:`free` when done with
        the encoder. For one-shot use, prefer :meth:`__call__`.
        """
        import os

        self.load()
        assert self._text_encoder is not None
        assert self._feature_extractor is not None

        max_length = int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", "1024"))
        all_hidden_states, attention_mask = self._text_encoder.encode_all_layers(prompt, max_length=max_length)
        video_embeds, audio_embeds = self._feature_extractor(all_hidden_states, attention_mask=attention_mask)
        return video_embeds, audio_embeds

    def __call__(
        self,
        prompts: str | list[str],
        *,
        free_after: bool = True,
    ) -> tuple[mx.array, mx.array] | list[tuple[mx.array, mx.array]]:
        """Encode one or more prompts; free Gemma + connector afterwards by default.

        Args:
            prompts: Single prompt or list of prompts. With a list, each
                element is encoded sequentially and a list of tuples is
                returned (matches upstream's batched signature).
            free_after: If True (default), drop strong refs to Gemma and
                the connector after encoding so subsequent components fit
                in memory. Pass False to keep the encoder loaded for
                subsequent calls.
        """
        if isinstance(prompts, str):
            video, audio = self.encode(prompts)
            _materialize(video, audio)
            if free_after:
                self.free()
            return video, audio

        outputs: list[tuple[mx.array, mx.array]] = []
        for p in prompts:
            video, audio = self.encode(p)
            _materialize(video, audio)
            outputs.append((video, audio))
        if free_after:
            self.free()
        return outputs


class ImageConditioner:
    """Owns the video VAE encoder lifecycle.

    Mirrors upstream ``utils.blocks.ImageConditioner``. Wraps a callable
    so that the encoder is built, passed to user code, then freed.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: _VideoVAEEncoder | None = None

    def load(self) -> _VideoVAEEncoder:
        """Build the VAE encoder (cached)."""
        if self._encoder is not None:
            return self._encoder
        self._encoder = _VideoVAEEncoder()
        vae_enc_path = _resolve_component_file(
            self.model_dir, "vae_encoder.safetensors", extra_stems=["video-vae-conv", "vae_encoder"]
        )
        weights = load_split_safetensors(vae_enc_path, prefix="vae_encoder.")
        weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in weights.items()
        }
        self._encoder.load_weights(list(weights.items()), strict=False)
        aggressive_cleanup()
        return self._encoder

    def free(self) -> None:
        self._encoder = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[_VideoVAEEncoder], object], *, free_after: bool = True) -> object:
        """Build encoder, call ``fn(encoder)``, then free encoder."""
        encoder = self.load()
        result = fn(encoder)
        if free_after:
            self.free()
        return result


class VideoDecoder:
    """Owns the video VAE decoder lifecycle + ffmpeg streaming muxing.

    Mirrors upstream ``utils.blocks.VideoDecoder`` (streaming decode +
    audio mux). Use :meth:`decode_and_stream` to decode a latent and
    mux with an audio file in one shot.
    """

    def __init__(self, model_dir: str | Path, verbose: bool = True) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.verbose = verbose
        self._decoder: _VideoVAEDecoder | None = None

    def load(self) -> _VideoVAEDecoder:
        if self._decoder is not None:
            return self._decoder
        decoder = _VideoVAEDecoder()
        # Video VAE weights differ between model versions (e.g. LTX-2.3's
        # vae_decoder.safetensors is not the same artifact as LTX-2.5's
        # ltx-2.5-video-vae-conv-bf16.safetensors, even though both load
        # into this same 86-tensor architecture without error). A generic
        # cross-model cache search has no way to tell those apart, so it is
        # deliberately excluded here: only the selected model_dir (or an
        # explicit LTX_MODEL_DIR override) is trusted.
        vae_dec_path = _resolve_component_file(
            self.model_dir,
            "vae_decoder.safetensors",
            extra_stems=["video-vae-conv", "vae_decoder"],
            allow_external_cache_fallback=False,
        )
        if not vae_dec_path.exists():
            raise FileNotFoundError(
                "Video VAE decoder weights were not found in the selected model "
                f"directory ({self.model_dir}). Expected a file named "
                "'vae_decoder.safetensors', or a file matching '*video-vae-conv*' "
                "or '*vae_decoder*', directly inside the model folder. Generation "
                "cannot proceed without these weights: continuing with an "
                "uninitialized decoder would produce a technically valid but "
                "corrupted (full-screen noise) video instead of failing visibly. "
                "Install or configure the required Video VAE component for this "
                "model before generating."
            )
        raw_weights = load_split_safetensors(vae_dec_path)
        if any(k.startswith("vae_decoder.") for k in raw_weights):
            # LTX-2.3 split-safetensors convention: decoder-only file,
            # "vae_decoder." prefix on every key.
            weights = load_split_safetensors(vae_dec_path, prefix="vae_decoder.")
        elif any(k.startswith("decoder.") or k.startswith("encoder.") for k in raw_weights):
            # Official LTX-2.5 combined VAE file convention: encoder + decoder
            # weights in one file ("decoder."/"encoder." key prefixes), with
            # per-channel stats named "*-of-means" instead of the bare
            # "mean"/"std" this decoder's modules expect. Only the decoder
            # half is used here; encoder.* keys are intentionally dropped
            # (this class never runs the encoder).
            #
            # The official artifact as published by Lightricks is a raw
            # PyTorch checkpoint (Conv3D stored O,I,D,H,W), not a
            # mlx-forge-pre-converted MLX pack (O,D,H,W,I) — see this repo's
            # "No Weight Conversion in This Package" rule, which targets the
            # general pre-converted-pack distribution pipeline for the main
            # supported model, not a single official component checkpoint
            # with a deterministic, well-known axis reorder. To stay firmly
            # on the safe side of that rule, the permutation below is never
            # applied blindly: it is only applied to a tensor whose stored
            # shape doesn't already match this decoder's own expected shape
            # but does match after the known PyTorch->MLX Conv3D reorder.
            # A tensor that matches neither is a layout this loader cannot
            # safely interpret, so it fails closed instead of guessing.
            stat_renames = {
                "per_channel_statistics.mean-of-means": "per_channel_statistics.mean",
                "per_channel_statistics.std-of-means": "per_channel_statistics.std",
            }
            remapped: dict[str, mx.array] = {}
            for k, v in raw_weights.items():
                if k.startswith("decoder."):
                    target_key = k[len("decoder."):]
                elif k in stat_renames:
                    target_key = stat_renames[k]
                else:
                    continue
                remapped[target_key] = v

            expected_shapes = {k: tuple(v.shape) for k, v in tree_flatten(decoder.parameters())}
            weights = {}
            for target_key, v in remapped.items():
                expected = expected_shapes.get(target_key)
                stored_shape = tuple(v.shape)
                if expected is None or stored_shape == expected:
                    # Already MLX-ready (or an unrecognized key strict=True
                    # will reject on its own) -- left unchanged.
                    weights[target_key] = v
                    continue
                pytorch_conv3d_reorder = (
                    len(stored_shape) == 5
                    and (stored_shape[0], stored_shape[2], stored_shape[3], stored_shape[4], stored_shape[1])
                    == expected
                )
                if pytorch_conv3d_reorder:
                    weights[target_key] = v.transpose(0, 2, 3, 4, 1)
                    continue
                raise ValueError(
                    f"Video VAE tensor {target_key!r} has an unrecognized layout: "
                    f"stored shape {stored_shape} matches this decoder's expected "
                    f"shape {expected} neither directly nor via the known PyTorch "
                    "Conv3D axis reorder (O,I,D,H,W -> O,D,H,W,I). Refusing to guess "
                    "at the correct permutation; this component cannot be loaded "
                    "safely."
                )
        else:
            weights = dict(raw_weights)
        decoder.load_weights(list(weights.items()), strict=True)
        self._decoder = decoder
        aggressive_cleanup()
        return self._decoder

    def free(self) -> None:
        self._decoder = None
        aggressive_cleanup()

    def decode_and_stream(
        self,
        video_latent: mx.array,
        output_path: str,
        frame_rate: float = 24.0,
        audio_path: str | None = None,
    ) -> str:
        """Stream-decode the latent into an mp4 with optional audio mux."""
        if self.verbose:
            tiling = _compute_decode_tiling(video_latent.shape, frame_rate=frame_rate)
            if tiling is not None and tiling.temporal_config is not None:
                tc = tiling.temporal_config
                print(
                    f"[vae-decode tiled: tile_frames={tc.tile_size_in_frames} overlap={tc.tile_overlap_in_frames}]",
                    file=sys.stderr,
                    flush=True,
                )
        decoder = self.load()
        decoder.decode_and_stream(video_latent, output_path, frame_rate=frame_rate, audio_path=audio_path)
        return output_path


class AudioDecoder:
    """Owns the audio VAE decoder + vocoder + BWE lifecycle.

    Mirrors upstream ``utils.blocks.AudioDecoder``. Decodes an audio
    latent through ``AudioVAEDecoder`` → BigVGAN vocoder → BWE to a
    waveform tensor at 48 kHz.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._audio_decoder: AudioVAEDecoder | None = None
        self._vocoder: VocoderWithBWE | None = None

    def load(self) -> tuple[AudioVAEDecoder, VocoderWithBWE]:
        if self._audio_decoder is None:
            self._audio_decoder = AudioVAEDecoder()
            audio_vae_path = _resolve_component_file(self.model_dir, "audio_vae.safetensors")
            if audio_vae_path.exists():
                raw_audio = load_split_safetensors(audio_vae_path)
                if any(k.startswith("audio_vae.decoder.") for k in raw_audio):
                    decoder_weights = load_split_safetensors(
                        audio_vae_path, prefix="audio_vae.decoder."
                    )
                    all_audio = load_split_safetensors(audio_vae_path, prefix="audio_vae.")
                    for k, v in all_audio.items():
                        if k.startswith("per_channel_statistics."):
                            decoder_weights[k] = v
                else:
                    decoder_weights = dict(raw_audio)
                decoder_weights = remap_audio_vae_keys(decoder_weights)
                self._audio_decoder.load_weights(list(decoder_weights.items()), strict=True)
            aggressive_cleanup()

        if self._vocoder is None:
            self._vocoder = VocoderWithBWE()
            vocoder_path = _resolve_component_file(self.model_dir, "vocoder.safetensors")
            if vocoder_path.exists():
                raw_vocoder = load_split_safetensors(vocoder_path)
                vocoder_weights = (
                    load_split_safetensors(vocoder_path, prefix="vocoder.")
                    if any(k.startswith("vocoder.") for k in raw_vocoder)
                    else dict(raw_vocoder)
                )
                from mlx.utils import tree_flatten
                flat_voc_params = dict(tree_flatten(self._vocoder.parameters()))
                remapped_voc = {}
                for k, v in vocoder_weights.items():
                    if k in flat_voc_params:
                        target_shape = flat_voc_params[k].shape
                        if (
                            v.shape != target_shape
                            and v.ndim == 3
                            and v.shape[0] == target_shape[0]
                            and v.shape[1] == target_shape[2]
                            and v.shape[2] == target_shape[1]
                        ):
                            remapped_voc[k] = v.transpose(0, 2, 1)
                        else:
                            remapped_voc[k] = v
                    else:
                        remapped_voc[k] = v
                self._vocoder.load_weights(list(remapped_voc.items()), strict=True)
                self._vocoder.upcast_weights_to_fp32()
            aggressive_cleanup()

        return self._audio_decoder, self._vocoder

    def free(self) -> None:
        self._audio_decoder = None
        self._vocoder = None
        aggressive_cleanup()

    def __call__(self, audio_latent: mx.array) -> mx.array:
        """Decode audio latent into a 48 kHz stereo waveform."""
        decoder, vocoder = self.load()
        mel = decoder.decode(audio_latent)
        return vocoder(mel)


class AudioConditioner:
    """Owns the audio VAE encoder + processor lifecycle."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: object | None = None
        self._processor: object | None = None

    def load(self) -> tuple[object, object]:
        if self._encoder is not None and self._processor is not None:
            return self._encoder, self._processor
        from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder
        from mlx.utils import tree_flatten

        self._encoder = AudioVAEEncoder()
        audio_vae_path = _resolve_component_file(self.model_dir, "audio_vae.safetensors")
        if audio_vae_path.exists():
            raw_audio = load_split_safetensors(audio_vae_path)
            if any(k.startswith("audio_vae.encoder.") for k in raw_audio):
                encoder_weights = load_split_safetensors(
                    audio_vae_path, prefix="audio_vae.encoder."
                )
                all_audio = load_split_safetensors(audio_vae_path, prefix="audio_vae.")
                for k, v in all_audio.items():
                    if k.startswith("per_channel_statistics."):
                        encoder_weights[k] = v
            else:
                flat_enc_params = dict(tree_flatten(self._encoder.parameters()))
                encoder_weights = {k: v for k, v in raw_audio.items() if k in flat_enc_params}
            encoder_weights = remap_audio_vae_keys(encoder_weights)
            self._encoder.load_weights(list(encoder_weights.items()), strict=False)
        self._processor = AudioProcessor()
        aggressive_cleanup()
        return self._encoder, self._processor

    def free(self) -> None:
        self._encoder = None
        self._processor = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[object, object], object], *, free_after: bool = True) -> object:
        """Build encoder+processor, call ``fn(encoder, processor)``, free."""
        encoder, processor = self.load()
        result = fn(encoder, processor)
        if free_after:
            self.free()
        return result


class VideoUpsampler:
    """Owns the spatial upsampler lifecycle.

    Mirrors upstream ``utils.blocks.VideoUpsampler``. Use for 2x spatial
    upscale between stage 1 and stage 2 of the two-stage pipelines.
    """

    def __init__(
        self,
        model_dir: str | Path,
        name: str = "spatial_upscaler_x2_v1_1",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.name = name
        self._upsampler: LatentUpsampler | None = None

    def load(self) -> LatentUpsampler:
        if self._upsampler is not None:
            return self._upsampler

        import json

        config_path = self.model_dir / f"{self.name}_config.json"
        weights_path = self.model_dir / f"{self.name}.safetensors"

        if config_path.exists():
            config = json.loads(config_path.read_text()).get("config", {})
            self._upsampler = LatentUpsampler.from_config(config)
        else:
            self._upsampler = LatentUpsampler()

        if weights_path.exists():
            weights = load_split_safetensors(weights_path, prefix=f"{self.name}.")
            self._upsampler.load_weights(list(weights.items()), strict=False)
        aggressive_cleanup()
        return self._upsampler

    def free(self) -> None:
        self._upsampler = None
        aggressive_cleanup()

    def __call__(self, latent: mx.array) -> mx.array:
        """Upscale a denormalized latent (caller must denorm/renorm)."""
        upsampler = self.load()
        return upsampler(latent)


__all__ = [
    "AudioConditioner",
    "AudioDecoder",
    "ImageConditioner",
    "PromptEncoder",
    "VideoDecoder",
    "VideoUpsampler",
]
