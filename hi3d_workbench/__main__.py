import argparse
import os
import sys
import threading
import webbrowser
from .common import config

def main():
    parser=argparse.ArgumentParser(description="Hi3D local workbench")
    parser.add_argument("--port",type=int,default=config()["port"])
    parser.add_argument("--open",action="store_true")
    args=parser.parse_args()
    from . import updater
    try:
        if updater.apply_on_start():
            os.execv(sys.executable, [sys.executable, '-m', 'hi3d_workbench', *sys.argv[1:]])
    except Exception as exc:
        if updater.recovery_required():
            raise RuntimeError('Interrupted update could not be restored; see data/updates/backups') from exc
        print(f'Code update was not installed; using existing program: {exc}', flush=True)
    if args.open:threading.Timer(2,lambda:webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    import uvicorn
    uvicorn.run("hi3d_workbench.server:app",host="127.0.0.1",port=args.port,access_log=False)

if __name__=="__main__":main()
