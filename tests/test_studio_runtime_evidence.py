import json

import pytest
import trimesh

from test_workbench import testroot
from hi3d_workbench.common import digest
from tools.record_studio_runtime import inspect_report


def actual_report(tmp_path,action='simplify',engine='meshoptimizer'):
    job=tmp_path/'data/jobs/actual';job.mkdir(parents=True)
    for name in ('coarse.glb','output.glb'):trimesh.creation.box().export(job/name)
    report={'status':'succeeded','mode':'studio','action':action,'engine':engine,
        'input_sha256':digest(job/'coarse.glb'),'output_sha256':digest(job/'output.glb'),
        'input_faces':12,'output_faces':12}
    path=job/'report.json';path.write_text(json.dumps(report))
    return path,report


def test_runtime_evidence_reads_actual_glb_and_rejects_changed_output(tmp_path):
    path,report=actual_report(tmp_path)
    assert inspect_report(path,'meshoptimizer',tmp_path)==report
    (path.parent/'output.glb').write_bytes(b'changed output')
    with pytest.raises(RuntimeError,match='哈希改变'):inspect_report(path,'meshoptimizer',tmp_path)


def test_runtime_evidence_rejects_false_face_counts_and_wrong_operation(tmp_path):
    path,report=actual_report(tmp_path)
    with pytest.raises(ValueError,match='操作不匹配'):inspect_report(path,'parts',tmp_path)
    report['output_faces']=999;path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='面数不匹配'):inspect_report(path,'meshoptimizer',tmp_path)


def test_rig_readiness_cannot_be_proven_by_static_glb(tmp_path):
    path,_=actual_report(tmp_path,'rig','skintokens')
    with pytest.raises(RuntimeError,match='实际蒙皮'):inspect_report(path,'skintokens',tmp_path)
