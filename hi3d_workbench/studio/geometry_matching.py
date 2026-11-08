"""Bounded, all-point Euclidean matching for X-Part's semantic features."""
from __future__ import annotations

import time
from types import MethodType


def nearest_indices(query, reference, *, matrix_elements=2_000_000, progress=lambda *args: None):
    """Return the first nearest reference index without a full B×Q×N matrix.

    Direct coordinate differences avoid the cancellation of the GEMM distance
    formula at coincident/very close surface samples. Every reference is tested;
    ties retain the first index, including across block boundaries.
    """
    import torch
    if (not isinstance(query, torch.Tensor) or not isinstance(reference, torch.Tensor)
            or query.ndim != 3 or reference.ndim != 3 or query.shape[-1] != 3 or reference.shape[-1] != 3
            or query.dtype != reference.dtype or query.dtype not in {torch.float32, torch.float64}
            or query.device != reference.device or not query.shape[0] or not reference.shape[0]
            or not reference.shape[1] or type(matrix_elements) is not int or matrix_elements < 1):
        raise ValueError('最近点匹配的点云、精度或内存预算无效')
    batches = max(query.shape[0], reference.shape[0])
    if query.shape[0] not in {1, batches} or reference.shape[0] not in {1, batches}:
        raise ValueError('最近点匹配的批次不兼容')
    with torch.no_grad():
        if not torch.isfinite(query).all() or not torch.isfinite(reference).all():
            raise ValueError('最近点匹配包含无效坐标')
        rows, columns = query.shape[1], reference.shape[1]
        result = torch.empty((batches, rows), dtype=torch.int64, device=query.device)
        if not rows:
            progress(0, 0)
            return result
        row_block = min(rows, 128, matrix_elements)
        column_block = min(columns, max(1, matrix_elements // row_block))
        for batch in range(batches):
            points = query[0 if query.shape[0] == 1 else batch].contiguous()
            targets = reference[0 if reference.shape[0] == 1 else batch].contiguous()
            for start in range(0, rows, row_block):
                current = points[start:start + row_block]
                best = torch.full((len(current),), float('inf'), dtype=query.dtype, device=query.device)
                indices = torch.zeros(len(current), dtype=torch.int64, device=query.device)
                for offset in range(0, columns, column_block):
                    distance = torch.cdist(current, targets[offset:offset + column_block], p=2,
                                           compute_mode='donot_use_mm_for_euclid_dist')
                    values, selected = distance.min(dim=-1)
                    closer = values < best
                    best = torch.where(closer, values, best)
                    indices = torch.where(closer, selected + offset, indices)
                if not torch.isfinite(best).all():
                    raise RuntimeError('最近点距离出现无效数值，已停止几何编码')
                result[batch, start:start + len(current)] = indices
                progress(batch * rows + start + len(current), batches * rows)
        return result


def install_xpart_conditioning(model, emit):
    """Retain the pinned full-batch encoders; bound only their nearest searches."""
    if (model.training or getattr(model, 'use_image', False)
            or not all(getattr(model, name, False) for name in ('use_geo', 'use_obj', 'use_seg_feat'))
            or not all(hasattr(model, name) for name in
                       ('geo_encoder', 'obj_encoder', 'seg_feat_encoder', 'geo_out_proj', 'obj_out_proj'))):
        raise RuntimeError('固定 X-Part 条件编码配置与有界语义匹配不兼容')

    def forward(self, part_surface_inbbox, object_surface):
        import torch
        context = {}
        context['geo_cond'], local_info = self.geo_encoder(part_surface_inbbox, object_surface,
                                                          return_local_pc_info=True)
        with torch.no_grad():
            context['obj_cond'], global_info = self.obj_encoder.encode_shape(object_surface, return_pc_info=True)
            point_features = self.seg_feat_encoder(object_surface[:1, ..., :3].float(),
                                                   object_surface[:1, ..., 3:6].float())
        self._hi3d_matching_report = []
        for name, info, projection, label in [
                ('obj_cond', global_info, self.obj_out_proj, '整体语义最近点匹配'),
                ('geo_cond', local_info, self.geo_out_proj, '局部语义最近点匹配')]:
            last = [0.]
            def progress(current, total):
                now = time.monotonic()
                if current == total or now - last[0] >= 1:
                    emit('asset_part_condition', label, device='cuda', progress_current=current,
                         progress_total=total, progress_unit='query points')
                    last[0] = now
            query, reference = info[0], object_surface[..., :3]
            matches = nearest_indices(query, reference, progress=progress)
            semantic = point_features.expand(part_surface_inbbox.shape[0], -1, -1).gather(
                1, matches.unsqueeze(-1).expand(-1, -1, point_features.size(-1)))
            context[name] = projection(torch.cat([context[name], semantic], dim=-1).to(projection.weight.dtype))
            rows, columns = query.shape[1], reference.shape[1]
            row_block = min(rows, 128, 2_000_000)
            block = row_block * min(columns, max(1, 2_000_000 // max(1, row_block)))
            self._hi3d_matching_report.append({'context': name, 'parts': part_surface_inbbox.shape[0],
                'queries_per_part': rows, 'reference_points_per_part': columns,
                'legacy_distance_matrix_bytes': matches.shape[0] * rows * columns * query.element_size(),
                'max_distance_block_bytes': block * query.element_size(),
                'precision': str(query.dtype), 'all_reference_points_tested': True,
                'distance_formula': 'direct coordinate differences; first-index ties'})
        return context

    model.forward = MethodType(forward, model)
