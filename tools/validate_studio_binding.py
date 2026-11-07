"""Independently read saved source/result GLBs and check binding preservation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hi3d_workbench.common import atomic_json, digest
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.skin import validate_preservation, validate_existing_rig


def verify(source, result, *, original_binding=None, contact_stage=None, existing_skeleton=False):
    """Validate original->binding->contact result without assuming 1:1 vertices."""
    source, result = Path(source), Path(result)
    if (original_binding is None) != (contact_stage is None):
        raise ValueError('接触分离校验需要同时指定原始绑定文件和接触阶段目录')
    document, output = Document.load(source), Document.load(result)
    raw = Document.load(original_binding) if original_binding is not None else output
    report = validate_preservation(document, raw)
    if existing_skeleton:
        report.update(validate_existing_rig(document, output))
    if original_binding is not None:
        from hi3d_workbench.studio.contact_skin import verify_saved_partition
        contact = verify_saved_partition(original_binding, result, contact_stage)
        report['contact_partition'] = contact
        report['original_binding_sha256'] = digest(original_binding)
        report['max_world_position_error'] += contact.get('max_world_triangle_error', 0.)
    report.update(source_sha256=digest(source), output_sha256=digest(result))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--original-binding', type=Path)
    parser.add_argument('--contact-stage', type=Path)
    parser.add_argument('--existing-skeleton', action='store_true')
    args = parser.parse_args()
    result = verify(args.source, args.result, original_binding=args.original_binding,
                    contact_stage=args.contact_stage, existing_skeleton=args.existing_skeleton)
    atomic_json(args.report, result)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
