import json
import hashlib
from pathlib import Path
import pytest

from test_workbench import testroot
from tools.package_studio_runtime import package,REQUIRED_FEATURES,REQUIRED_NOTICES,sha,validate_notices
from hi3d_workbench.updater import owned_name


def fixture(tmp_path):
    source,pipeline=tmp_path/'development',tmp_path/'bundle/runtime-pipeline'
    source.mkdir();pipeline.mkdir(parents=True)
    files=['meshoptimizer.dll','autoremesher/autoremesher.exe','Lib/site-packages/example.py',
        'engines/kimodo-v510/model.py','engines/unirig/model.py',
        'sources/parts/LICENSE','sources/skintokens/LICENSE','sources/unirig/LICENSE','sources/kimodo/LICENSE',
        'licenses/NOTICE','blender/blender.exe']
    for name in files:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    core=source/'autoremesher/core';core.mkdir()
    executable=core/'autoremesher-core.exe';executable.write_bytes(b'console-core')
    dlls={}
    for name in ('msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll'):
        (core/name).write_bytes(name.encode());dlls[name]=sha(core/name)
    (core/'build-receipt.json').write_text(json.dumps({
        'source_revision':'4a49f82f720f647691c09a7d8a16fa63be208aad',
        'sha256':sha(executable),'runtime_dlls':dlls}))
    notices=source/'licenses/asset-studio';notices.mkdir()
    (notices/'LICENSE.txt').write_bytes(b'upstream license fixture')
    entry={'path':'LICENSE.txt','bytes':(notices/'LICENSE.txt').stat().st_size,'sha256':sha(notices/'LICENSE.txt')}
    (notices/'manifest.json').write_text(json.dumps({'schema':1,'components':[
        {'component':name,'upstream':'https://example.org/upstream','revision':'pinned-fixture','files':[entry]}
        for name in sorted(REQUIRED_NOTICES)]}))
    source_components={}
    for name in ('parts','skintokens','unirig','kimodo'):
        payload=(source/'sources'/name/'LICENSE').read_bytes()
        source_components[name]={'revision':'0'*40,'files':{'LICENSE':{'bytes':len(payload),
            'sha1':hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()}}}
    (source/'sources/source-manifest.json').write_text(json.dumps({'schema':1,'components':source_components}))
    for name in ['fpsample-build.json','motion-correction-build.json']:
        (source/name).write_text(json.dumps({'sha256':'fixture','build_project':'E:/developer/private'}))
    for name in ['python.exe','python312.dll','python3.dll','python312.zip','DLLs/example.pyd','Lib/site-packages/shared.py']:
        path=pipeline/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    evidence=source/'validation/runtime';evidence.mkdir(parents=True)
    features={}
    for name in REQUIRED_FEATURES:
        report={'status':'succeeded','mode':'studio','action':name,'engine':name,'input_sha256':'a'*64,'output_sha256':'b'*64}
        path=evidence/(name+'.json');path.write_text(json.dumps(report))
        features[name]={'ready':True,'validation_reports':[{'path':path.relative_to(source).as_posix(),
            'sha256':sha(path),**{key:report[key] for key in ('action','engine','input_sha256','output_sha256')}}]}
    features['meshoptimizer']['sha256']=sha(source/'meshoptimizer.dll')
    state={'features':features,'blender':str(source/'blender/blender.exe'),
        'blender_sha256':sha(source/'blender/blender.exe'),
        'source_manifest_sha256':sha(source/'sources/source-manifest.json'),
        'notice_manifest_sha256':sha(source/'licenses/asset-studio/manifest.json')}
    (source/'studio-runtime.json').write_text(json.dumps(state))
    return source,pipeline,tmp_path/'bundle/runtime-assets'


def test_runtime_package_relocates_shared_python_blender_and_skips_stale_overlay(tmp_path):
    source,pipeline,target=fixture(tmp_path)
    (source/'engines/kimodo').mkdir();(source/'engines/kimodo/stale.txt').write_text('obsolete')
    result=package(source,pipeline,target)
    assert result==target
    assert not (target/'engines/kimodo').exists()
    assert (target/'engines/kimodo-v510/model.py').is_file()
    paths=(target/'python312._pth').read_text().splitlines()
    assert '../runtime-pipeline/Lib/site-packages' in paths
    assert all(not Path(path).is_absolute() for path in paths)
    assert json.loads((target/'studio-runtime.json').read_text())['blender']=='blender/blender.exe'
    assert 'build_project' not in json.loads((target/'motion-correction-build.json').read_text())
    assert (source/'engines/kimodo/stale.txt').read_text()=='obsolete'


def test_unverified_runtime_cannot_create_a_release_directory(tmp_path):
    source,pipeline,target=fixture(tmp_path)
    state=json.loads((source/'studio-runtime.json').read_text());state['features']['kimodo']['ready']=False
    (source/'studio-runtime.json').write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match='kimodo'):package(source,pipeline,target)
    assert not target.exists()


def test_runtime_package_refuses_overwrite_and_code_update_stays_code_only(tmp_path):
    source,pipeline,target=fixture(tmp_path);target.mkdir();(target/'personal.txt').write_text('keep')
    with pytest.raises(ValueError,match='已存在'):package(source,pipeline,target)
    assert (target/'personal.txt').read_text()=='keep'
    assert owned_name('runtime_requirements/studio-kimodo.txt')
    with pytest.raises(ValueError):owned_name('runtime_requirements/model.safetensors')
    with pytest.raises(ValueError):owned_name('runtime-assets/python.exe')


def test_runtime_package_rejects_lost_or_modified_notices_before_creating_output(tmp_path):
    source,pipeline,target=fixture(tmp_path)
    (source/'licenses/asset-studio/LICENSE.txt').write_bytes(b'modified notice')
    with pytest.raises(RuntimeError,match='校验失败'):package(source,pipeline,target)
    assert not target.exists()


def test_notice_manifest_cannot_read_outside_its_directory(tmp_path):
    source,_,_=fixture(tmp_path);folder=source/'licenses/asset-studio'
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest['components'][0]['files'][0]['path']='../NOTICE'
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='路径无效'):validate_notices(folder)


def test_runtime_package_checks_pinned_source_bytes_and_rejects_extra_code(tmp_path):
    source,pipeline,target=fixture(tmp_path)
    code=source/'sources/parts/model.py';code.write_bytes(b'unrecorded code')
    with pytest.raises(RuntimeError,match='未记录文件'):package(source,pipeline,target)
    assert not target.exists()
    code.unlink()
    (source/'sources/parts/LICENSE').write_bytes(b'changed tracked source')
    with pytest.raises(RuntimeError,match='源码校验失败'):package(source,pipeline,target)
    assert not target.exists()


def test_package_uses_verified_console_core_without_legacy_qt_files(tmp_path):
    source,pipeline,target=fixture(tmp_path)
    core=source/'autoremesher/core'
    (source/'autoremesher/Qt5Core.dll').write_bytes(b'legacy-qt')
    package(source,pipeline,target)
    assert (target/'autoremesher/autoremesher-core.exe').read_bytes()==b'console-core'
    assert not (target/'autoremesher/autoremesher.exe').exists()
    assert not (target/'autoremesher/Qt5Core.dll').exists()
    assert (target/'licenses/asset-studio/manifest.json').is_file()
    (core/'vcruntime140.dll').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='运行库'):package(source,pipeline,target.parent/'broken')


def test_legacy_qt_gui_cannot_satisfy_the_console_distribution_gate(tmp_path):
    source,pipeline,target=fixture(tmp_path)
    (source/'autoremesher/core/autoremesher-core.exe').unlink()
    assert (source/'autoremesher/autoremesher.exe').is_file()
    with pytest.raises(RuntimeError,match='不能打包旧 Qt GUI'):package(source,pipeline,target)
    assert not target.exists()


def test_ready_flag_does_not_replace_portable_runtime_evidence(tmp_path):
    source,pipeline,target=fixture(tmp_path)
    (source/'validation/runtime/parts.json').write_text('{}')
    with pytest.raises(RuntimeError,match='运行证据'):package(source,pipeline,target)
    assert not target.exists()
