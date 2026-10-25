"""Report completed official sampler steps without replacing the sampler."""
from __future__ import annotations

from functools import wraps
import inspect


def track_sampler(sampler, emit, phase, message):
    original_sample = sampler.sample
    invocation = 0

    @wraps(original_sample)
    def sample(*args, **kwargs):
        nonlocal invocation
        invocation += 1
        params = inspect.signature(original_sample).bind_partial(*args, **kwargs)
        params.apply_defaults()
        total = int(params.arguments['steps'])
        current = 0
        original_once = sampler.sample_once
        previous_override = sampler.__dict__.get('sample_once')
        had_override = 'sample_once' in sampler.__dict__

        def progress():
            emit('sampling', phase=phase, device='cuda', step=current,
                 num_inference_steps=total, pass_index=invocation-1,
                 message=f'{message} · {current}/{total}')

        @wraps(original_once)
        def once(*a, **kw):
            nonlocal current
            result = original_once(*a, **kw)
            current += 1
            progress()
            return result

        sampler.sample_once = once
        try:
            progress()
            return original_sample(*args, **kwargs)
        finally:
            if had_override:
                sampler.sample_once = previous_override
            else:
                del sampler.sample_once

    sampler.sample = sample
