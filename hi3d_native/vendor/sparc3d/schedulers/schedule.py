"""Shared flow-matching schedule construction and effective guidance."""

# Written for SPARC3D against the measured behavior of the released weights,
# not copied from an upstream sampler. The rational timestep rescale and the
# ``1000 * t`` model time originate with Microsoft TRELLIS; the token-count
# warp, the two-level gamma, and the guidance interval are specific to these
# checkpoints. Changing any of these constants changes the output geometry.

from __future__ import annotations

from numbers import Integral, Real

import torch
from torch import Tensor

DEFAULT_NUM_INFERENCE_STEPS = 25
"""Integrator updates run per pass unless the caller asks for another count."""

SIGMA_MIN = 0.002
"""Terminal flow noise scale the schedule descends to."""

TIME_RESCALE_FACTOR = 2.0
"""Rational rescale applied to base timesteps before the token-count warp."""


def effective_guidance_scale(timestep: float | Tensor) -> float | Tensor:
    """Return the effective zero-negative guidance scale at ``timestep``.

    Args:
        timestep: Current model time after schedule warping and sigma inflation.

    Returns:
        Scale 4 at the first time, ``10 * timestep**2`` on ``[0.7, 1)``,
        and 1 below 0.7.
    """
    if isinstance(timestep, Tensor):
        one = torch.ones_like(timestep)
        return torch.where(
            timestep >= 1.0,
            one * 4.0,
            torch.where(timestep >= 0.7, 10.0 * timestep.square(), one),
        )

    time = float(timestep)
    if time >= 1.0:
        return 4.0
    if time >= 0.7:
        return 10.0 * time**2
    return 1.0


def apply_effective_guidance(
    positive_prediction: Tensor,
    negative_prediction: Tensor,
    timestep: float | Tensor,
) -> Tensor:
    """Combine positive and zero-condition predictions with effective guidance.

    Args:
        positive_prediction: Conditional model prediction.
        negative_prediction: Model prediction under the all-zero condition tensor.
        timestep: Current model time after schedule warping and sigma inflation.

    Returns:
        Guided prediction, or the unchanged positive prediction below time 0.7.

    Raises:
        ValueError: Positive and negative predictions have different shapes.
    """
    if positive_prediction.shape != negative_prediction.shape:
        raise ValueError(
            "positive and negative predictions must have the same shape, "
            f"got {list(positive_prediction.shape)} and "
            f"{list(negative_prediction.shape)}"
        )
    scale = effective_guidance_scale(timestep)
    if isinstance(scale, Tensor):
        scale = scale.to(
            device=positive_prediction.device, dtype=positive_prediction.dtype
        )
    return negative_prediction + scale * (
        positive_prediction - negative_prediction
    )


class FlowMatchSchedule:
    """Token-dependent trajectory and update bookkeeping shared by the integrators."""

    order = 1
    """Flow-velocity evaluations one completed update consumes."""

    def __init__(
        self,
        *,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        sigma_min: float = SIGMA_MIN,
        time_rescale_factor: float = TIME_RESCALE_FACTOR,
    ) -> None:
        """Configure schedule defaults without constructing a token-dependent schedule.

        Args:
            num_inference_steps: Number of updates used by default.
            sigma_min: Terminal flow noise scale.
            time_rescale_factor: Rational rescale applied before token warping.

        Raises:
            ValueError: A numeric configuration value is outside its valid range.
        """
        self.num_inference_steps = self._validate_steps(num_inference_steps)
        if not isinstance(sigma_min, Real) or not 0.0 <= float(sigma_min) < 1.0:
            raise ValueError(f"sigma_min must be in [0, 1), got {sigma_min!r}")
        if (
            not isinstance(time_rescale_factor, Real)
            or float(time_rescale_factor) <= 0.0
        ):
            raise ValueError(
                "time_rescale_factor must be positive, "
                f"got {time_rescale_factor!r}"
            )

        self.sigma_min = float(sigma_min)
        self.time_rescale_factor = float(time_rescale_factor)
        self.token_warp_factor: float | None = None
        self.base_timesteps: Tensor | None = None
        self.warped_timesteps: Tensor | None = None
        self.sigmas: Tensor | None = None
        self.gammas: Tensor | None = None
        self.sigma_hats: Tensor | None = None
        self.timesteps: Tensor | None = None
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        num_tokens: int,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """Construct the effective schedule for a sparse token count.

        Args:
            num_inference_steps: Number of updates; ``None`` uses the configured
                default.
            num_tokens: Number of sparse tokens in the current sample.
            device: Device on which to create schedule tensors.

        Returns:
            Inflated model times for each update.

        Raises:
            ValueError: ``num_inference_steps`` or ``num_tokens`` is invalid.
        """
        steps = (
            self.num_inference_steps
            if num_inference_steps is None
            else self._validate_steps(num_inference_steps)
        )
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, Integral):
            raise ValueError(f"num_tokens must be a positive integer, got {num_tokens!r}")
        if int(num_tokens) <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens!r}")

        self.num_inference_steps = steps
        token_count = int(num_tokens)
        warp = (19.0 * token_count - 20000.0) / (
            3.0 * token_count + 60000.0
        )
        self.token_warp_factor = max(1.0, min(5.0, warp))

        base = torch.linspace(
            1.0, 0.0, steps + 1, dtype=torch.float32, device=device
        )
        rescaled = self._rational_rescale(base, self.time_rescale_factor)
        warped = self._rational_rescale(rescaled, self.token_warp_factor)
        sigmas = self.sigma_min + (1.0 - self.sigma_min) * warped
        gammas = torch.where(
            warped[:-1] > 0.985,
            torch.full_like(warped[:-1], 0.04),
            torch.full_like(warped[:-1], 0.002),
        )
        sigma_hats = sigmas[:-1] * (1.0 + gammas)

        self.base_timesteps = base
        self.warped_timesteps = warped
        self.sigmas = sigmas
        self.gammas = gammas
        self.sigma_hats = sigma_hats
        self.timesteps = sigma_hats
        self._step_index = 0
        return self.timesteps

    def _pending_update(
        self, model_output: Tensor, timestep: float | Tensor, sample: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Validate the update ``timestep`` names and measure how far it travels.

        The schedule is left where it is, so a predictor and its corrector can both
        resolve the same pending update.

        Args:
            model_output: Flow velocity predicted at ``timestep``.
            timestep: Current value from :attr:`timesteps`.
            sample: Current noisy sample.

        Returns:
            Step size down to the next uninflated sigma, and that sigma.

        Raises:
            RuntimeError: :meth:`set_timesteps` was not called or the schedule is
                exhausted.
            ValueError: ``timestep`` is not the next scheduled model time, or the
                tensors have different shapes.
        """
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("set_timesteps must be called before step")
        if self._step_index >= self.num_inference_steps:
            raise RuntimeError("scheduler step called after the schedule was exhausted")
        if model_output.shape != sample.shape:
            raise ValueError(
                "model_output and sample must have the same shape, "
                f"got {list(model_output.shape)} and {list(sample.shape)}"
            )

        expected = self.timesteps[self._step_index]
        actual = torch.as_tensor(timestep, device=expected.device, dtype=expected.dtype)
        if actual.numel() != 1 or not torch.isclose(
            actual.reshape(()), expected, rtol=1e-5, atol=1e-6
        ):
            raise ValueError(
                f"expected timestep {expected.item():.8g} at step "
                f"{self._step_index}, got {timestep!r}"
            )

        next_sigma = self.sigmas[self._step_index + 1]
        step_size = (expected - next_sigma).to(
            device=sample.device, dtype=sample.dtype
        )
        return step_size, next_sigma

    @staticmethod
    def _rational_rescale(timestep: Tensor, factor: float) -> Tensor:
        return factor * timestep / (1.0 + (factor - 1.0) * timestep)

    @staticmethod
    def _validate_steps(num_inference_steps: int) -> int:
        if (
            isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, Integral)
            or int(num_inference_steps) <= 0
        ):
            raise ValueError(
                "num_inference_steps must be a positive integer, "
                f"got {num_inference_steps!r}"
            )
        return int(num_inference_steps)
