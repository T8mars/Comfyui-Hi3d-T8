import argparse
import shutil
from pathlib import Path
from build_release import ROOT,ALLOWED,build_source,manifest

p=argparse.ArgumentParser();p.add_argument('release',type=Path);args=p.parse_args()
release=args.release.resolve()
if not release.is_relative_to(ROOT/'dist') or not (release/'runtime/python.exe').is_file():
    raise SystemExit('只能更新本项目 dist 内已有的独立发行目录')
for name in ALLOWED:
    source=ROOT/name
    if source.is_dir():shutil.copytree(source,release/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    elif source.is_file():
        (release/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,release/name)
build_source();manifest()
print(release)
