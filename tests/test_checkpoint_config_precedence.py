"""Config discovery fallback for ``LTXModelConfig.from_checkpoint_dir``.

A checkpoint directory can carry two JSON config sources: ``embedded_config.json``
(richer, historically preferred — see
``test_av_ca_timestep_config.py::test_from_checkpoint_dir_prefers_embedded``)
and ``config.json`` (always carries a top-level ``model_version``). Some
third-party-packaged checkpoints ship BOTH, where ``embedded_config.json`` is a
stale artifact from an older packaging pipeline: it parses as valid JSON but
has no top-level ``model_version`` and therefore fails strict checkpoint
validation.

Before this fix, a candidate that parsed as JSON but failed that semantic
validation raised a bare ``ValueError`` straight out of ``from_checkpoint_dir``
and never gave the next candidate a chance — so a checkpoint with a perfectly
valid ``config.json`` sitting right next to a stale ``embedded_config.json``
failed to load entirely, even though a usable config was one file away. These
tests pin the fallback: try the next candidate on either a JSON-parse failure
or a semantic-validation failure, and only fail once every candidate
(including the GGUF-embedded-config path) has been exhausted.

Fixture names are neutral by design (``legacy_style_config`` /
``versioned_config``) — this fallback is purely structural (file shape and
validity), never keyed on any model name or filesystem path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltx_core_mlx.model.transformer.model import LTXModelConfig

_TRANSFORMER = {
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "num_layers": 48,
    "cross_attention_dim": 4096,
}

# Shaped like a genuinely valid config.json: flat, top-level model_version.
_VERSIONED_CONFIG = {"model_version": "2.3.0", "transformer": _TRANSFORMER}

# Shaped like the real-world stale embedded_config.json this fix targets:
# valid JSON, nested per-component sub-dicts (transformer/vae/scheduler), but
# no top-level model_version anywhere in the document.
_LEGACY_UNVERSIONED_CONFIG = {
    "transformer": _TRANSFORMER,
    "vae": {"some": "vae-config"},
    "scheduler": {"some": "scheduler-config"},
}

# Shaped like a real-world third-party-published config.json this fix also
# targets: no "transformer" wrapper key at all — the transformer fields sit
# directly at the top level, alongside model_version.
_FLAT_VERSIONED_CONFIG = {"model_version": "2.3.0", **_TRANSFORMER}


# 1. Only a valid config.json exists.
def test_only_versioned_config_json_is_used(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 3)
    assert cfg.num_layers == 48


# 2. Only a valid embedded_config.json exists (no config.json fallback needed).
def test_only_valid_embedded_config_is_used(tmp_path: Path):
    (tmp_path / "embedded_config.json").write_text(json.dumps(_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 3)
    assert cfg.num_layers == 48


# 3. Both exist, the preferred candidate (embedded_config.json) is valid.
#    Existing precedence is untouched: a VALID embedded_config.json still
#    wins over config.json. This fix only changes what happens when the
#    preferred candidate FAILS validation (test below) — it must not change
#    what happens when it succeeds.
def test_valid_embedded_config_still_wins_over_config_json(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_version": "2.3.0", "transformer": {**_TRANSFORMER, "num_layers": 1}})
    )
    (tmp_path / "embedded_config.json").write_text(json.dumps(_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.num_layers == 48  # from embedded_config.json, not config.json's 1


# 4. THE KEY REGRESSION TEST for this fix. Both exist, but the preferred
#    candidate is the real-world stale shape (parses fine, no model_version).
#    Before this fix this raised and never reached config.json; now it must
#    fall through and succeed using config.json.
def test_stale_embedded_config_falls_back_to_valid_config_json(tmp_path: Path):
    (tmp_path / "embedded_config.json").write_text(json.dumps(_LEGACY_UNVERSIONED_CONFIG))
    (tmp_path / "config.json").write_text(json.dumps(_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 3)
    assert cfg.num_layers == 48


# 5. Both invalid (neither declares model_version) -> fails clearly, no
#    silent default and no silent choice of either file.
def test_both_invalid_fails_clearly(tmp_path: Path):
    (tmp_path / "embedded_config.json").write_text(json.dumps(_LEGACY_UNVERSIONED_CONFIG))
    (tmp_path / "config.json").write_text(json.dumps({"transformer": _TRANSFORMER}))
    with pytest.raises(ValueError, match="model_version"):
        LTXModelConfig.from_checkpoint_dir(tmp_path)


# A JSON parse failure (not just a semantic one) on the preferred candidate
# must also fall through, not only a semantically-invalid-but-parseable one.
def test_malformed_json_also_falls_through(tmp_path: Path):
    (tmp_path / "embedded_config.json").write_text("{not json")
    (tmp_path / "config.json").write_text(json.dumps(_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 3)


# 9. No model-specific string matching: the fallback is purely structural
#    (file shape and validity). An arbitrarily-named directory must resolve
#    identically to any other directory with the same file shapes.
def test_no_model_specific_string_matching(tmp_path: Path):
    named_dir = tmp_path / "some-third-party-checkpoint-name"
    named_dir.mkdir()
    (named_dir / "embedded_config.json").write_text(json.dumps(_LEGACY_UNVERSIONED_CONFIG))
    (named_dir / "config.json").write_text(json.dumps(_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(named_dir)
    assert cfg.model_version == (2, 3)


# 6. Canonical LTX-2.3 config layout remains accepted.
def test_ltx23_config_layout_accepted(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"model_version": "2.3.0", "transformer": _TRANSFORMER}))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 3)


# 7. Current LTX-2.5 versioned-config layout remains accepted.
def test_ltx25_style_versioned_config_layout_accepted(tmp_path: Path):
    t25 = {
        **_TRANSFORMER,
        "ff_bias": False,
        "audio_ff_bias": True,
        "use_prompt_adaln_single": True,
        "use_keyframes_abs_pos_embedding": True,
    }
    config = {
        "model_version": "2.5.0",
        "transformer": t25,
        "gemma_source_checkpoint": {"gemma_version": "gemma4-12b-ltx-v1"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 5)
    assert cfg.ff_bias is False


# 10. Failure diagnostics stay informative: the raised error names the
#     directory searched and preserves the underlying reason, rather than
#     collapsing to a generic message that hides which candidate almost
#     worked.
def test_failure_diagnostics_mention_directory_and_reason(tmp_path: Path):
    (tmp_path / "embedded_config.json").write_text(json.dumps(_LEGACY_UNVERSIONED_CONFIG))
    with pytest.raises(ValueError) as excinfo:
        LTXModelConfig.from_checkpoint_dir(tmp_path)
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "model_version" in message


# Unchanged: no candidates at all -> FileNotFoundError, not silent defaults.
def test_missing_everything_still_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no transformer config"):
        LTXModelConfig.from_checkpoint_dir(tmp_path)


# Unchanged: strict=False opt-out still returns hardcoded defaults instead of
# raising, even when the only candidate present is the stale/invalid shape.
def test_legacy_opt_out_still_returns_defaults_on_invalid_config(tmp_path: Path):
    (tmp_path / "embedded_config.json").write_text(json.dumps(_LEGACY_UNVERSIONED_CONFIG))
    assert LTXModelConfig.from_checkpoint_dir(tmp_path, strict=False) == LTXModelConfig()


# A second, independent gap found while proving this fix against a real
# third-party package: validate_strict_checkpoint_contract() required a
# nested "transformer" sub-dict unconditionally, while from_checkpoint_config's
# own field extraction already accepted a bare (wrapper-less) transformer
# dict via `config.get("transformer", config)` (test_accepts_bare_transformer_dict
# in test_av_ca_timestep_config.py). A flat config.json — transformer fields
# directly at the top level, next to model_version, no "transformer" key —
# parsed fine, passed field extraction, and then failed strict validation
# anyway. Fixed by applying the identical fallback in the validator.
def test_flat_config_without_transformer_wrapper_key_is_accepted(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(_FLAT_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 3)
    assert cfg.num_layers == 48


def test_stale_embedded_config_falls_back_to_flat_config_json(tmp_path: Path):
    """Both gaps together: the exact real-world shape this fix was built for.

    embedded_config.json parses but has no model_version (gap 1); config.json
    has model_version but no "transformer" wrapper key (gap 2). Both must be
    bridged for the checkpoint to load at all.
    """
    (tmp_path / "embedded_config.json").write_text(json.dumps(_LEGACY_UNVERSIONED_CONFIG))
    (tmp_path / "config.json").write_text(json.dumps(_FLAT_VERSIONED_CONFIG))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.model_version == (2, 3)
    assert cfg.num_layers == 48
