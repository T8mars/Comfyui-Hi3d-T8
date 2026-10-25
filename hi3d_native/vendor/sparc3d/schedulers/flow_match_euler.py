"""First-order Euler integrator over the shared flow-matching schedule."""

from __future__ import annotations

from torch import Tensor

from sparc3d.schedulers.schedule import FlowMatchSchedule


class FlowMatchEulerScheduler(FlowMatchSchedule):
    """Plain first-order scheduler for the released flow-matching trajectory."""

    order = 1

    def step(
        self, model_output: Tensor, timestep: float | Tensor, sample: Tensor
    ) -> Tensor:
        """Advance one first-order Euler update without churn noise.

        Args:
            model_output: Flow velocity predicted at ``timestep``.
            timestep: Current value from :attr:`timesteps`.
            sample: Current noisy sample.

        Returns:
            Sample advanced to the next uninflated sigma.

        Raises:
            RuntimeError: :meth:`set_timesteps` was not called or the schedule is
                exhausted.
            ValueError: ``timestep`` is not the next scheduled model time, or the
                tensors have different shapes.
        """
        step_size, _ = self._pending_update(model_output, timestep, sample)
        previous_sample = sample - step_size * model_output
        self._step_index += 1
        return previous_sample
