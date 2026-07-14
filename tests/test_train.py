"""Tests for train.py's pure helpers and the difficulty-overshoot knob.

These functions steer the whole learning run (what counts as fitness, how
hard the tracks get), so their arithmetic deserves the same scrutiny as the
simulation itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from racing.config import TrackConfig
from racing.track import _knob, make_track
from train import aggregate_fitness, champion_key

# fitness of 4 genomes (columns) on 3 tracks (rows)
PER_TRACK = np.array([
    [1.0, 5.0, 2.0, 0.0],
    [2.0, 1.0, 2.0, 0.0],
    [9.0, 3.0, 2.0, 3.0],
], dtype=np.float32)


def test_aggregate_mean_min():
    """'mean' and 'min' are the two poles: average-case vs worst-case scoring."""
    np.testing.assert_allclose(aggregate_fitness(PER_TRACK, "mean", 0.5),
                               [4.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(aggregate_fitness(PER_TRACK, "min", 0.5),
                               [1.0, 1.0, 2.0, 0.0])


def test_aggregate_cvar_is_mean_of_worst_tracks():
    """CVaR = mean of the worst ceil(frac*K) tracks: with K=3 and frac=0.5
    that is the worst 2 — pressure against tail-event crashes that plain
    mean-fitness never applies."""
    np.testing.assert_allclose(aggregate_fitness(PER_TRACK, "cvar", 0.5),
                               [1.5, 2.0, 2.0, 0.0])
    # frac=1.0 degrades gracefully to the mean; tiny frac to the min.
    np.testing.assert_allclose(aggregate_fitness(PER_TRACK, "cvar", 1.0),
                               aggregate_fitness(PER_TRACK, "mean", 0.5))
    np.testing.assert_allclose(aggregate_fitness(PER_TRACK, "cvar", 0.01),
                               aggregate_fitness(PER_TRACK, "min", 0.5))


def test_aggregate_unknown_raises():
    with pytest.raises(ValueError):
        aggregate_fitness(PER_TRACK, "median", 0.5)


def test_knob_piecewise_interpolation():
    """The difficulty knob runs easy->hard over [0,1] and hard->extreme over
    (1, max]: curriculum overshoot must extrapolate smoothly, and a config
    with max_difficulty=1.0 must never touch the extreme values."""
    assert _knob(50.0, 22.0, 18.0, 0.0, 1.3) == 50.0
    assert _knob(50.0, 22.0, 18.0, 1.0, 1.3) == 22.0
    assert _knob(50.0, 22.0, 18.0, 1.3, 1.3) == pytest.approx(18.0)
    assert _knob(50.0, 22.0, 18.0, 1.15, 1.3) == pytest.approx(20.0)
    assert _knob(50.0, 22.0, 18.0, 0.5, 1.3) == pytest.approx(36.0)
    # no overshoot configured -> clamps at hard
    assert _knob(50.0, 22.0, 18.0, 1.2, 1.0) == 22.0


def test_champion_key_noise_min_falls_through_to_mean():
    """Early in training every candidate's val_min is quantization noise
    (everyone crashes on the hardest validation tracks within a few px);
    a noise-level min difference must NOT outrank a multi-lap mean
    difference, or champion selection degenerates into a random tiebreak."""
    noisy_but_great = champion_key(0.015, 4.77)
    lucky_min_but_awful = champion_key(0.035, 0.23)
    assert noisy_but_great > lucky_min_but_awful


def test_champion_key_material_min_improvement_dominates():
    """A min improvement of a full bucket (0.25 laps) is a real robustness
    gain and must outrank any mean difference — that is the whole point of
    worst-case selection."""
    robust = champion_key(1.30, 6.0)
    fragile_fast = champion_key(0.90, 9.5)
    assert robust > fragile_fast
    # Within the same bucket, mean decides.
    assert champion_key(8.50, 10.5) > champion_key(8.60, 8.9)


def test_make_track_clips_to_max_difficulty():
    """Requesting past the ceiling must clamp (and report the clamped value),
    not extrapolate into geometry the validity checks never vetted."""
    cfg = TrackConfig()
    t = make_track(31, 99.0, cfg, 6.0)
    assert t.difficulty == pytest.approx(cfg.max_difficulty)
