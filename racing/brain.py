"""The cars' brains: a tiny fixed-topology neural network, stored flat.

Each car's brain is a two-layer perceptron:

    inputs (7 ray distances + speed)  ->  hidden (tanh)  ->  2 outputs (tanh)
                                                              [steer, throttle]

The genome IS the network: all weights and biases flattened into one float32
vector of length G. The whole population is a single (P, G) matrix — the
central data structure of the project. The genetic algorithm never needs to
know it's mutating a neural network; it just perturbs rows of a matrix.

The forward pass is *batched across brains*, not just across data: every car
has different weights, so we keep a (P, in, hidden) weight tensor and contract
it with the (P, in) observations in one einsum. Two einsums evaluate all 512
different networks simultaneously — this is the "vectorize across models"
trick that makes neuroevolution cheap on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BrainConfig, SensorConfig

N_OUTPUTS = 2  # [steer, throttle], both in [-1, 1] via tanh


@dataclass(frozen=True)
class BrainSpec:
    """Resolved network shape + the genome slice boundaries derived from it."""
    n_in: int
    hidden: int
    n_out: int = N_OUTPUTS

    @property
    def genome_size(self) -> int:
        return self.n_in * self.hidden + self.hidden + self.hidden * self.n_out + self.n_out


def make_spec(brain_cfg: BrainConfig, sensor_cfg: SensorConfig) -> BrainSpec:
    # +1 input: the car's own normalized speed (proprioception).
    return BrainSpec(n_in=len(sensor_cfg.ray_angles_deg) + 1, hidden=brain_cfg.hidden)


def unpack(genomes: np.ndarray, spec: BrainSpec):
    """Zero-copy views of the flat (P, G) genome matrix as weight tensors."""
    p = genomes.shape[0]
    i, h, o = spec.n_in, spec.hidden, spec.n_out
    a = i * h
    b = a + h
    c = b + h * o
    w1 = genomes[:, :a].reshape(p, i, h)
    b1 = genomes[:, a:b]
    w2 = genomes[:, b:c].reshape(p, h, o)
    b2 = genomes[:, c:]
    return w1, b1, w2, b2


def forward(genomes: np.ndarray, obs: np.ndarray, spec: BrainSpec) -> np.ndarray:
    """Evaluate every car's own network on its own observation.

    genomes: (P, G) float32, obs: (P, n_in) float32 -> controls (P, 2) in [-1, 1].
    'pi,pih->ph' means: for each population member p, contract its input
    vector with its personal weight matrix — a batched matrix-vector product.
    """
    w1, b1, w2, b2 = unpack(genomes, spec)
    h = np.tanh(np.einsum("pi,pih->ph", obs, w1) + b1)
    return np.tanh(np.einsum("ph,pho->po", h, w2) + b2)


def init_population(pop: int, spec: BrainSpec, rng: np.random.Generator,
                    scale: float = 1.0) -> np.ndarray:
    """Random initial genomes, weights ~ N(0, scale/sqrt(fan_in)), biases 0.

    The 1/sqrt(fan_in) scaling (Xavier-style) keeps tanh pre-activations in
    their responsive range at generation 0, so initial behavior is *diverse*
    rather than every net saturated at ±1 doing the same thing.
    """
    genomes = np.empty((pop, spec.genome_size), dtype=np.float32)
    w1, b1, w2, b2 = unpack(genomes, spec)
    w1[:] = rng.normal(0.0, scale / np.sqrt(spec.n_in), w1.shape)
    b1[:] = 0.0
    w2[:] = rng.normal(0.0, scale / np.sqrt(spec.hidden), w2.shape)
    b2[:] = 0.0
    return genomes
