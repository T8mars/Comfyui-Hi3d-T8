"""Download selected optional engines serially, resuming an existing download."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from filelock import Timeout
from hi3d_workbench import pipeline_models


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('selections',nargs='+')
    args=parser.parse_args()
    for selection in args.selections:
        pipeline_models.selection_components(selection)
        waiting=False
        while True:
            try:
                result=pipeline_models.download(selection)
                print('ENGINE_DOWNLOAD_COMPLETE',selection,flush=True)
                break
            except Timeout:
                if not waiting:
                    print('ENGINE_DOWNLOAD_WAITING_FOR_ACTIVE_LOCK',selection,flush=True)
                    waiting=True
                time.sleep(5)


if __name__=='__main__':main()
