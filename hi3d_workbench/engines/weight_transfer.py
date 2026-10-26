"""Bounded CPU-to-CUDA staging, without dtype changes or whole-model pinning.

Two pinned slabs alternate between synchronous CPU staging and asynchronous
DMA on one private CUDA stream. A slab is never overwritten until its event
has completed. All copies finish before returning or propagating an error.
"""
import os
import sys
import time

CHUNK_BYTES = 16 * 2**20
MIN_BYTES = 64 * 2**20


def move(value, device, *args, emit=None, label='模型', mode=None, **kwargs):
    """Use bounded staging for supported large modules, otherwise ordinary to()."""
    torch = sys.modules.get('torch')
    mode = mode or os.environ.get('HI3D_WEIGHT_TRANSFER', 'auto')
    if mode not in {'auto', 'on', 'off'}:
        raise ValueError('Invalid HI3D_WEIGHT_TRANSFER; use auto, on or off')
    began = time.perf_counter()
    target = str(device)
    cuda = target.startswith('cuda')

    def report(message, **fields):
        if emit:
            emit('loading', phase='model_transfer', device='cuda' if cuda else 'cpu', message=message, **fields)

    def ordinary(reason):
        report(f'{label}：使用标准权重搬运', transfer_backend='standard', transfer_reason=reason)
        result = value.to(device, *args, **kwargs)
        report(f'{label}：权重搬运完成（{time.perf_counter()-began:.1f} 秒）',
               transfer_backend='standard', model_transfer_seconds=round(time.perf_counter()-began, 4), transfer_reason=reason)
        return result

    if mode == 'off' or not cuda or torch is None or args or kwargs:
        return ordinary('disabled, non-CUDA, or explicit conversion arguments')
    module = getattr(value, 'model', value)
    if not hasattr(torch, 'nn') or not isinstance(module, torch.nn.Module):
        return ordinary('custom module')
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors or any(t.device.type != 'cpu' or t.layout != torch.strided or
                          t.is_quantized or not t.is_contiguous() or type(t) not in {torch.Tensor, torch.nn.Parameter}
                          for t in tensors):
        return ordinary('requires ordinary tensor/device conversion')
    unique = {id(t): t for t in tensors}
    total = sum(t.numel() * t.element_size() for t in unique.values())
    if total < MIN_BYTES:
        return ordinary('small module')
    import psutil
    if psutil.virtual_memory().available < 512 * 2**20 + 2 * CHUNK_BYTES:
        return ordinary('host memory guard')
    target = torch.device(device)
    free, _ = torch.cuda.mem_get_info(target)
    reusable = torch.cuda.memory_reserved(target) - torch.cuda.memory_allocated(target)
    if total + 256 * 2**20 > free + reusable:
        return ordinary('device memory guard')

    # Allocation failure is safe to fall back only before any model mutation
    # or CUDA work has been enqueued. Never retry after a CUDA execution error.
    try:
        with torch.inference_mode(False), torch.no_grad():
            slabs = [torch.empty(CHUNK_BYTES, dtype=torch.uint8, device='cpu', pin_memory=True) for _ in range(2)]
    except (RuntimeError, MemoryError) as error:
        return ordinary('pinned allocation unavailable: ' + str(error).splitlines()[0])

    stream = torch.cuda.Stream(device=target)
    slots = [None, None]
    sizes = [0, 0]
    converted = {}
    finished = 0
    queued = 0
    chunks = 0
    staging_seconds = 0.0
    last_report = began
    report(f'{label}：分块读取并上传显存', transfer_backend='pinned_chunks', progress_current=0,
           progress_total=total, progress_unit='bytes', pinned_bytes=2 * CHUNK_BYTES)

    def convert(tensor):
        nonlocal finished, queued, chunks, staging_seconds, last_report
        previous = converted.get(id(tensor))
        if previous is not None:
            return previous
        if tensor.device.type != 'cpu' or not tensor.numel():
            return tensor.to(target)
        destination = torch.empty_like(tensor, device=target)
        source_flat, target_flat = tensor.view(-1), destination.view(-1)
        count = max(1, CHUNK_BYTES // tensor.element_size())
        for offset in range(0, tensor.numel(), count):
            slot = chunks % 2
            if slots[slot] is not None:
                slots[slot].synchronize()
                finished += sizes[slot]
            size = min(count, tensor.numel()-offset)
            host = slabs[slot].view(tensor.dtype)[:size]
            copied = time.perf_counter()
            host.copy_(source_flat[offset:offset+size])
            staging_seconds += time.perf_counter()-copied
            target_flat[offset:offset+size].copy_(host, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
            slots[slot], sizes[slot] = event, size * tensor.element_size()
            queued += sizes[slot]
            chunks += 1
            if time.perf_counter()-last_report >= .5:
                report(f'{label}：已上传 {finished/2**20:.0f}/{total/2**20:.0f} MiB',
                       transfer_backend='pinned_chunks', progress_current=finished,
                       progress_total=total, progress_unit='bytes')
                last_report = time.perf_counter()
        converted[id(tensor)] = destination
        return destination

    try:
        with torch.cuda.device(target), torch.cuda.stream(stream), torch.inference_mode(False), torch.no_grad():
            module._apply(convert)
    finally:
        # Keep host slabs and GPU allocations alive until DMA finishes, even
        # after a failed _apply. CPU source chunks were staged synchronously.
        stream.synchronize()
    result = value.to(device)  # Preserve wrapper/custom to() side effects.
    report(f'{label}：显存权重已就绪（{time.perf_counter()-began:.1f} 秒）',
           transfer_backend='pinned_chunks', model_transfer_seconds=round(time.perf_counter()-began, 4),
           model_staging_seconds=round(staging_seconds, 4), transfer_chunks=chunks,
           pinned_bytes=2 * CHUNK_BYTES, progress_current=queued, progress_total=total, progress_unit='bytes')
    return result
