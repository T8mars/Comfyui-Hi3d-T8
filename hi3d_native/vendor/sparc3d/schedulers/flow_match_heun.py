"""Second-order Heun integrator over the shared flow-matching schedule."""

from __future__ import annotations

from torch import Tensor

from sparc3d.schedulers.schedule import FlowMatchSchedule


class FlowMatchHeunScheduler(FlowMatchSchedule):
    """Predictor-corrector scheduler that averages both slopes of an update.

    An update is split so the caller can evaluate the flow between the two halves:
    :meth:`predict` returns the Euler predictor and the time its slope belongs to,
    and :meth:`step` consumes that second slope. Both halves resolve the same
    pending update, and only :meth:`step` advances the schedule.
    """

    order = 2

    def predict(
        self, model_output: Tensor, timestep: float | Tensor, sample: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Take the Euler predictor of the pending update.

        Args:
            model_output: Flow velocity predicted at ``timestep``.
            timestep: Current value from :attr:`timesteps`.
            sample: Current noisy sample.

        Returns:
            Predicted sample at the next uninflated sigma, and that sigma. The
            final update predicts at :attr:`sigma_min`.

        Raises:
            RuntimeError: :meth:`set_timesteps` was not called or the schedule is
                exhausted.
            ValueError: ``timestep`` is not the next scheduled model time, or the
                tensors have different shapes.
        """
        step_size, next_sigma = self._pending_update(model_output, timestep, sample)
        return sample - step_size * model_output, next_sigma

    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        *,
        second_output: Tensor,
    ) -> Tensor:
        """Advance one second-order Heun update without churn noise.

        Args:
            model_output: Flow velocity predicted at ``timestep``.
            timestep: Current value from :attr:`timesteps`.
            sample: Current noisy sample, not the predicted one.
            second_output: Flow velocity predicted on the :meth:`predict` sample at
                the next uninflated sigma. Keyword-only and required so a
                first-order caller cannot silently drop the second slope.

        Returns:
            Sample advanced to the next uninflated sigma along the averaged slope.

        Raises:
            RuntimeError: :meth:`set_timesteps` was not called or the schedule is
                exhausted.
            ValueError: ``timestep`` is not the next scheduled model time, or the
                tensors have different shapes.
        """
        step_size, _ = self._pending_update(model_output, timestep, sample)
        if second_output.shape != sample.shape:
            raise ValueError(
                "second_output and sample must have the same shape, "
                f"got {list(second_output.shape)} and {list(sample.shape)}"
            )
        averaged_output = 0.5 * (model_output + second_output)
        previous_sample = sample - step_size * averaged_output
        self._step_index += 1
        return previous_sample
