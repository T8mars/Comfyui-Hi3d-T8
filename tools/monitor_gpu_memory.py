"""Validation-only NVML sampler including native CUDA allocations.

Records whole-device memory, including desktop/other processes; it is not a
measurement of one process or proof of a minimum supported GPU capacity.
"""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hi3d_workbench.common import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-file',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=900)
    parser.add_argument('--interval',type=float,default=.1)
    args=parser.parse_args()
    if args.output.exists() or args.stop_file.exists():parser.error('use fresh output and stop paths')
    if not .05<=args.interval<=10 or not 0<args.seconds<=14400:parser.error('invalid sample interval/duration')
    if os.name=='nt':
        dll=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/nvml.dll'
        api=ctypes.CDLL(str(dll))
    else:api=ctypes.CDLL('libnvidia-ml.so.1')
    def check(code):
        if code:raise RuntimeError(f'NVML error {code}')
    class Memory(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    check(api.nvmlInit_v2())
    handle=ctypes.c_void_p()
    check(api.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(handle)))
    samples=[];started=time.monotonic();last_write=started
    def save(done):
        atomic_json(args.output,{'metric':'NVML whole-device memory; includes driver, desktop and other processes',
                    'completed':done,'interval_seconds':args.interval,'elapsed_seconds':round(time.monotonic()-started,3),
                    'baseline_mb':samples[0][1] if samples else None,
                    'peak_used_mb':max((point[1] for point in samples),default=None),'samples':samples})
    try:
        while True:
            memory=Memory();check(api.nvmlDeviceGetMemoryInfo(handle,ctypes.byref(memory)))
            samples.append([round(time.monotonic()-started,3),round(memory.used/2**20,3)])
            if args.stop_file.exists() or time.monotonic()-started>=args.seconds:break
            if time.monotonic()-last_write>=5:
                save(False);last_write=time.monotonic()
            time.sleep(args.interval)
    finally:
        save(True);api.nvmlShutdown()
    print({'peak_used_mb':max(point[1] for point in samples),'samples':len(samples)},flush=True)


if __name__=='__main__':main()
