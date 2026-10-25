"""Shared UV preparation and auditable phase timings for workbench painters."""
from __future__ import annotations
import time


def prepare(mesh, request, emit):
    from ..common import DATA
    from .mesh_uv import ensure_uv
    start=time.perf_counter()
    phases=[]
    details={}
    reasons=[]
    current=None
    boundary=start

    def event(**values):
        nonlocal current,boundary
        now=time.perf_counter()
        identity=(values.get('phase'),values.get('device'))
        if identity!=current:
            if current is not None:
                phases.append({'phase':current[0],'device':current[1],'seconds':round(now-boundary,3)})
            current,boundary=identity,now
        for key in ('cache_hit','cache_key','uv_backend','removed_degenerate_faces',
                    'local_uv_repaired_faces','packing_backend','numba_pack','uv_repair','uv_diagnostics','packing_fallback_reason'):
            if key in values:details[key]=values[key]
        reason=values.get('fallback_reason') or values.get('packing_fallback_reason')
        if reason and reason not in reasons:reasons.append(reason)
        emit(values.pop('stage','unwrapping'),**values)

    result,policy=ensure_uv(mesh,emit=event,cache_dir=DATA/'uv-cache',
                            texture_size=int(request.get('texture_size',2048)))
    end=time.perf_counter()
    if current is not None:
        phases.append({'phase':current[0],'device':current[1],'seconds':round(end-boundary,3)})
    return result,policy,{'seconds':round(end-start,3),'policy':policy,'phases':phases,
                         'details':details,'fallback_reasons':reasons}
