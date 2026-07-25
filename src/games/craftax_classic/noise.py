"""NumPy Perlin / fractal noise for Craftax-Classic worldgen.

Adapted from Pierre Vigier's perlin-numpy (same source as the JAX Craftax
port under ``reference/craftax/.../util/noise.py``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def interpolant(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def generate_perlin_noise_2d(
    rng: np.random.Generator,
    shape: tuple[int, int],
    res: tuple[int, int],
    interpolant_fn: Callable[[np.ndarray], np.ndarray] = interpolant,
) -> np.ndarray:
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2.0 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    t = interpolant_fn(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2.0) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
    rng: np.random.Generator,
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: int = 2,
    interpolant_fn: Callable[[np.ndarray], np.ndarray] = interpolant,
) -> np.ndarray:
    noise = np.zeros(shape, dtype=np.float64)
    frequency = 1
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            rng,
            shape,
            (frequency * res[0], frequency * res[1]),
            interpolant_fn,
        )
        frequency *= lacunarity
        amplitude *= persistence

    noise_min = float(noise.min())
    noise_max = float(noise.max())
    denom = noise_max - noise_min
    if denom <= 1e-12:
        return np.zeros(shape, dtype=np.float64)
    return (noise - noise_min) / denom
