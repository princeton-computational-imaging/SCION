"""Tests for config composition and strategy YAML auto-loading.

Verifies that:
  - Strategy YAMLs are loaded automatically based on model.*_strategy values
  - codebook.yaml overrides take precedence over strategy YAML defaults
  - Missing strategy YAMLs fail loudly (no silent fallback)
  - Full config composition produces expected parameter values
"""

import pytest
import yaml
from pathlib import Path

from scion.models.codebook import CodebookModelConfig
from scion.strategies import CodebookStrategy, MHSplatStrategy
from scion.util.hydra import CONF_DIR, apply_strategy_defaults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_codebook_config_with_strategies(model_overrides: dict = None) -> CodebookModelConfig:
    """Build a CodebookModelConfig with strategy YAML auto-loading.

    Strategy YAMLs provide defaults, model_overrides take precedence.
    """
    model_cfg_dict = apply_strategy_defaults(dict(model_overrides or {}))

    # Filter to only valid CodebookModelConfig fields
    valid_fields = {f.name for f in CodebookModelConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in model_cfg_dict.items() if k in valid_fields}

    return CodebookModelConfig(**filtered)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStrategyYamlAutoLoading:
    """Strategy YAMLs are loaded based on model.*_strategy values."""

    def test_splat_mh_params_loaded(self):
        """splat_strategy='mh' loads birth proposal params from splat/mh.yaml."""
        cfg = build_codebook_config_with_strategies({"splat_strategy": "mh"})

        assert cfg.splat_birth_accept_threshold == 0.5
        assert cfg.splat_min_opacity == 0.01

    def test_splat_mh_timing_loaded(self):
        """splat_strategy='mh' loads timing overrides from splat/mh.yaml."""
        cfg = build_codebook_config_with_strategies({"splat_strategy": "mh"})

        assert cfg.splat_refine_start == 500
        assert cfg.splat_refine_every == 100
        assert cfg.splat_refine_stop == 25000


class TestStrategyYamlOverride:
    """Explicit model keys override strategy YAML defaults."""

    def test_explicit_override_takes_precedence(self):
        """A key in the model dict overrides the strategy YAML value."""
        cfg = build_codebook_config_with_strategies({
            "splat_strategy": "mh",
            "splat_birth_accept_threshold": 0.7,  # Override the 0.5 from mh.yaml
        })

        assert cfg.splat_birth_accept_threshold == 0.7, (
            "Explicit override should take precedence over strategy YAML"
        )

    def test_timing_override(self):
        """Explicit splat_refine_every overrides strategy YAML timing."""
        cfg = build_codebook_config_with_strategies({
            "splat_strategy": "mh",
            "splat_refine_every": 50,  # Override the 100 from mh.yaml
        })

        assert cfg.splat_refine_every == 50


class TestMissingStrategyYaml:
    """Missing strategy YAMLs fail loudly."""

    def test_nonexistent_strategy_raises(self):
        """A nonexistent strategy name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            build_codebook_config_with_strategies({
                "splat_strategy": "nonexistent_strategy_xyz"
            })


class TestComposedConfig:
    """Full composition produces expected values."""

    def test_canonical_codebook_stack(self):
        """Canonical codebook config: MH splat / default instance."""
        cfg = build_codebook_config_with_strategies({
            "splat_strategy": "mh",
            "instance_strategy": "default",
        })

        # Splat timing from splat/mh.yaml
        assert cfg.splat_refine_start == 500
        assert cfg.splat_refine_every == 100

        # MH splat birth params from splat/mh.yaml
        assert cfg.splat_birth_accept_threshold == 0.5
        assert cfg.splat_min_opacity == 0.01


class TestNoStrategyDefaultsInCodebookYaml:
    """Regression: codebook.yaml defaults should not list strategy entries."""

    def test_no_strategy_defaults(self):
        """codebook.yaml defaults list has no /strategies/ entries."""
        codebook_path = Path(CONF_DIR) / "method" / "codebook.yaml"
        with open(codebook_path) as f:
            raw = yaml.safe_load(f)

        defaults = raw.get("defaults", [])
        for entry in defaults:
            if isinstance(entry, dict):
                for key in entry:
                    assert "/strategies/" not in key and "strategies/" not in key, (
                        f"codebook.yaml should not have strategy defaults, found: {key}"
                    )
            elif isinstance(entry, str):
                assert "strategies" not in entry, (
                    f"codebook.yaml should not have strategy defaults, found: {entry}"
                )


class _UserSplatStrategy(CodebookStrategy):
    """Loaded by import path in TestStrategyImportPath."""


class _MHSubclass(MHSplatStrategy):
    """A trivial built-in subclass: inherits config_name = "mh"."""


class TestStrategyImportPath:
    """``splat_strategy=pkg.mod:Class`` loads a user strategy class."""

    def _populate(self, splat, instance):
        from test_dispatch_wired import _make_tiny_codebook_model

        model = _make_tiny_codebook_model()
        model.config.splat_strategy = splat
        model.config.instance_strategy = instance
        model._populate_strategies(model.num_candidates, model.num_instances, model.T_max)
        return model.codebook_strategy

    def test_import_path_loads_user_class(self):
        composite = self._populate(f"{__name__}:_UserSplatStrategy", "none")
        assert isinstance(composite.splat, _UserSplatStrategy)
        assert composite.instances == []

    def test_builtin_names_still_resolve(self):
        from scion.strategies import DefaultInstanceStrategy, MHSplatStrategy

        composite = self._populate("mh", "default")
        assert isinstance(composite.splat, MHSplatStrategy)
        assert isinstance(composite.instances[0], DefaultInstanceStrategy)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="splat_strategy='nonexistent_strategy_xyz'"):
            self._populate("nonexistent_strategy_xyz", "none")

    def test_bad_import_path_names_the_key(self):
        with pytest.raises(ValueError, match="instance_strategy="):
            self._populate("mh", "no_such_pkg.mod:Cls")

    def test_non_strategy_class_rejected(self):
        with pytest.raises(TypeError, match="not a CodebookStrategy subclass"):
            self._populate("collections:OrderedDict", "none")

    def test_builtin_subclass_keeps_yaml_defaults(self):
        timing = ("splat_refine_start", "splat_refine_every", "splat_refine_stop",
                  "splat_min_opacity", "splat_birth_accept_threshold")
        mh = apply_strategy_defaults({"splat_strategy": "mh"})
        sub = apply_strategy_defaults({"splat_strategy": f"{__name__}:_MHSubclass"})
        assert {k: sub[k] for k in timing} == {k: mh[k] for k in timing}

    def test_plain_custom_strategy_loads_no_yaml(self):
        cfg = apply_strategy_defaults({
            "splat_strategy": f"{__name__}:_UserSplatStrategy", "instance_strategy": "none",
        })
        assert cfg == {"splat_strategy": f"{__name__}:_UserSplatStrategy", "instance_strategy": "none"}
