import math
from typing import Literal

import torch

from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS

CHROMA_SCHEDULER_NAME_VALUES = Literal["euler", "euler_cfg_pp_beta", "heun", "lcm"]

CHROMA_SCHEDULER_LABELS: dict[str, str] = {
    **FLUX_SCHEDULER_LABELS,
    "euler_cfg_pp_beta": "Euler CFG++ (Beta)",
}

# Chroma follows FLUX sampling in the reference implementation. Beta(0.6, 0.6)
# quantiles select from a 10,000-point float32 ModelSamplingFlux table shifted by 1.15.
# Keeping the inverse-CDF implementation local avoids adding SciPy solely for this schedule.
_BETA_ALPHA = 0.6
_BETA_BETA = 0.6
_FLUX_TIMESTEPS = 10000
_FLUX_SHIFT = 1.15


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized incomplete beta function."""
    max_iterations = 200
    epsilon = 3e-14
    fp_min = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fp_min:
        d = fp_min
    d = 1.0 / d
    h = d

    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + doubled) * (a + doubled))
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        h *= d * c

        aa = -(a + iteration) * (qab + iteration) * x / ((a + doubled) * (qap + doubled))
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break

    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    beta_term = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return beta_term * _beta_continued_fraction(a, b, x) / a
    return 1.0 - beta_term * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_ppf(probability: float, a: float = _BETA_ALPHA, b: float = _BETA_BETA) -> float:
    """Inverse Beta CDF with enough precision for stable discrete timestep selection."""
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0

    # The preset is symmetric. Reflecting the upper half keeps the
    # bisection away from the numerically steep x->1 endpoint.
    if math.isclose(a, b) and probability > 0.5:
        return 1.0 - _beta_ppf(1.0 - probability, a=a, b=b)

    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) * 0.5
        if _regularized_incomplete_beta(midpoint, a, b) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) * 0.5


def get_chroma_beta_schedule(num_steps: int) -> list[float]:
    """Return the reference Beta(0.6, 0.6) schedule used for Chroma.

    Quantiles select from the same 10,000-point float32 FLUX sigma table used by
    the reference implementation. Consecutive duplicate indices are dropped and
    a terminal zero is appended. The local inverse-CDF avoids a SciPy dependency.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero")

    total_timestep_index = _FLUX_TIMESTEPS - 1
    shift = math.exp(_FLUX_SHIFT)
    sigmas: list[float] = []
    last_timestep_index = -1
    for step_index in range(num_steps):
        probability = 1.0 - step_index / num_steps
        quantile = _beta_ppf(probability)
        timestep_index = int(round(quantile * total_timestep_index))
        if timestep_index != last_timestep_index:
            # ModelSamplingFlux builds this table in torch.float32 from
            # torch.arange(1, 10001) / 10000, then applies the Flux time shift.
            timestep = torch.tensor(
                (timestep_index + 1) / _FLUX_TIMESTEPS,
                dtype=torch.float32,
            )
            sigma = shift / (shift + (1.0 / timestep - 1.0))
            sigmas.append(float(sigma.item()))
        last_timestep_index = timestep_index

    sigmas.append(0.0)
    return sigmas
