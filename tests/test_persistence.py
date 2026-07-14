"""Tests for racing/persistence.py: genome checkpoints (.npz) and metrics CSV.

Persistence is where silent corruption hides: a genome that loads with a
different dtype, a Config field that comes back as a list instead of a tuple,
or a CSV whose columns drift will all *look* fine until an experiment quietly
becomes irreproducible. These tests pin down bit-exact round-trips.
"""

from __future__ import annotations

import csv
import dataclasses

import numpy as np
import pytest

from racing.config import Config, SensorConfig, TrainConfig
from racing.persistence import MetricsLogger, load_genome, save_genome


def _custom_config() -> Config:
    """A Config with non-default values so round-trip tests can't pass by
    accidentally reconstructing defaults instead of reading the file."""
    return Config(
        seed=7,
        sensor=SensorConfig(ray_angles_deg=(-30.0, 0.0, 30.0),
                            ray_length=120.0, n_samples=12),
        train=dataclasses.replace(TrainConfig(),
                                  val_seeds=(101, 202, 303),
                                  val_difficulties=(0.25, 0.75)),
    )


# ---------------------------------------------------------------------------
# save_genome / load_genome
# ---------------------------------------------------------------------------

def test_genome_roundtrip_bit_identical(tmp_path):
    """The genome IS the trained artifact — any bit flip changes the policy.
    A checkpoint must return exactly the float32 values that were saved."""
    rng = np.random.default_rng(0)
    genome = rng.normal(size=178).astype(np.float32)
    path = str(tmp_path / "champ.npz")

    save_genome(path, genome, Config())
    loaded, _, _ = load_genome(path)

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, genome)  # bit-identical, no tolerance


def test_genome_saved_as_float32_even_from_float64(tmp_path):
    """save_genome normalizes to float32 so the on-disk format is stable no
    matter what dtype the caller happened to be holding."""
    rng = np.random.default_rng(1)
    genome64 = rng.normal(size=32)  # float64
    path = str(tmp_path / "champ64.npz")

    save_genome(path, genome64, Config())
    loaded, _, _ = load_genome(path)

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, genome64.astype(np.float32))


def test_config_roundtrips_through_checkpoint(tmp_path):
    """A genome without its exact Config is meaningless (different ray layout
    = different observation space), so the checkpoint must rebuild the Config
    perfectly — including tuple-typed fields that JSON degrades to lists."""
    cfg = _custom_config()
    genome = np.zeros(10, dtype=np.float32)
    path = str(tmp_path / "cfg.npz")

    save_genome(path, genome, cfg)
    _, cfg2, _ = load_genome(path)

    # Full structural equality (frozen dataclasses compare by value).
    assert cfg2 == cfg

    # Spot-check scalars across subsystems.
    assert cfg2.seed == 7
    assert cfg2.sensor.ray_length == 120.0
    assert cfg2.sensor.n_samples == 12
    assert cfg2.car.dt == pytest.approx(1.0 / 30.0)
    assert cfg2.evo.population == 512

    # Tuple fields must come back as tuples, not JSON lists — downstream code
    # relies on hashability/immutability of frozen-dataclass fields.
    assert isinstance(cfg2.sensor.ray_angles_deg, tuple)
    assert cfg2.sensor.ray_angles_deg == (-30.0, 0.0, 30.0)
    assert isinstance(cfg2.train.val_seeds, tuple)
    assert cfg2.train.val_seeds == (101, 202, 303)
    assert isinstance(cfg2.train.val_difficulties, tuple)
    assert cfg2.train.val_difficulties == (0.25, 0.75)


def test_meta_scalars_survive_roundtrip(tmp_path):
    """Meta carries provenance (generation number, fitness). It must return
    as plain Python scalars with values intact, keyed without the prefix."""
    genome = np.ones(4, dtype=np.float32)
    path = str(tmp_path / "meta.npz")

    save_genome(path, genome, Config(), meta={"gen": 42, "fitness": 3.75})
    _, _, meta = load_genome(path)

    assert set(meta) == {"gen", "fitness"}
    assert meta["gen"] == 42
    assert isinstance(meta["gen"], int)       # .item() -> native Python int
    assert meta["fitness"] == 3.75            # exactly representable in binary
    assert isinstance(meta["fitness"], float)


def test_no_meta_gives_empty_dict(tmp_path):
    """Omitting meta is the common case; it must load as {} not crash."""
    path = str(tmp_path / "nometa.npz")
    save_genome(path, np.zeros(3, dtype=np.float32), Config())
    _, _, meta = load_genome(path)
    assert meta == {}


def test_save_genome_creates_parent_directories(tmp_path):
    """Checkpoints are written mid-training into runs/<name>/ folders that
    may not exist yet; save must not require callers to mkdir first."""
    path = str(tmp_path / "runs" / "exp1" / "best.npz")
    save_genome(path, np.zeros(3, dtype=np.float32), Config())
    genome, cfg, _ = load_genome(path)
    assert genome.shape == (3,)
    assert cfg == Config()


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

def test_metrics_logger_writes_header_once(tmp_path):
    """Re-attaching a logger to an existing file (training resume) must not
    inject a second header row into the middle of the data."""
    path = str(tmp_path / "metrics.csv")
    MetricsLogger(path)
    MetricsLogger(path)  # resume: file already exists

    with open(path, newline="") as f:
        lines = f.read().splitlines()
    assert len(lines) == 1
    assert lines[0].split(",") == MetricsLogger.COLUMNS


def test_metrics_logger_appends_rows_readable_by_dictreader(tmp_path):
    """Rows must land under the right column names so any CSV tool (pandas,
    DictReader) reads the numbers back correctly mid-training."""
    path = str(tmp_path / "metrics.csv")
    logger = MetricsLogger(path)
    logger.log(gen=0, best_fit=1.5, sigma=0.2)
    logger.log(gen=1, best_fit=2.25, sigma=0.199)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["gen"] == "0"
    assert rows[0]["best_fit"] == "1.5"
    assert rows[1]["gen"] == "1"
    assert float(rows[1]["sigma"]) == 0.199


def test_metrics_logger_missing_columns_are_empty_cells(tmp_path):
    """Validation columns only exist every val_every generations; on other
    rows they must be empty cells (honest gaps), not zeros or stale values."""
    path = str(tmp_path / "metrics.csv")
    logger = MetricsLogger(path)
    logger.log(gen=0, best_fit=1.0)  # no val_* columns this generation

    with open(path, newline="") as f:
        row = next(csv.DictReader(f))

    assert row["val_mean"] == ""
    assert row["val_min"] == ""
    assert row["val_gap"] == ""
    # And every cell exists — the row spans all columns.
    assert set(row) == set(MetricsLogger.COLUMNS)


def test_metrics_logger_unknown_column_raises(tmp_path):
    """A typo like best_fitt would silently vanish under row.get(); the
    explicit ValueError turns that data-loss bug into an immediate crash."""
    logger = MetricsLogger(str(tmp_path / "metrics.csv"))
    with pytest.raises(ValueError, match="unknown metric columns"):
        logger.log(gen=0, best_fitt=1.0)


def test_metrics_logger_unknown_column_writes_nothing(tmp_path):
    """The validation happens before the file is opened, so a bad log call
    must leave the CSV untouched (header only)."""
    path = str(tmp_path / "metrics.csv")
    logger = MetricsLogger(path)
    with pytest.raises(ValueError):
        logger.log(nope=1)
    with open(path, newline="") as f:
        assert len(f.read().splitlines()) == 1  # header only


# ---------------------------------------------------------------------------
# Config JSON round-trip (used by the checkpoint format)
# ---------------------------------------------------------------------------

def test_config_json_roundtrip_default():
    """The default Config must survive to_json -> from_json exactly; this is
    the invariant the whole checkpoint format leans on."""
    cfg = Config()
    assert Config.from_json(cfg.to_json()) == cfg


def test_config_json_roundtrip_customized():
    """Non-default values (including nested and tuple fields) must round-trip
    too — otherwise only the trivial case works."""
    cfg = _custom_config()
    cfg2 = Config.from_json(cfg.to_json())
    assert cfg2 == cfg
    assert isinstance(cfg2.train.val_seeds, tuple)
    assert isinstance(cfg2.sensor.ray_angles_deg, tuple)
