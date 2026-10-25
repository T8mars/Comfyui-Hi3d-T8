import argparse
import os
import sys
import threading
from .common import ROOT, VERSION, config, resolve_model_dir

def main():
    parser=argparse.ArgumentParser(description="Hi3D local workbench")
    parser.add_argument("--port",type=int,default=config()["port"])
    parser.add_argument("--open",action="store_true")
    parser.add_argument("--no-browser",dest="open",action="store_false")
    parser.add_argument("--stop",action="store_true",help="Stop only the workbench in this directory")
    args=parser.parse_args()
    if not 1 <= args.port <= 65535:parser.error('port must be between 1 and 65535')
    from . import console_control, lifecycle
    result=console_control.existing(args.port,stop=args.stop,open_browser=args.open and not args.stop)
    if result is not None:return result
    from . import updater
    print(f'[启动] Hi3D 工作台 v{VERSION}', flush=True)
    print(f'[启动] 工作台目录：{ROOT}', flush=True)
    print(f'[启动] 模型目录：{resolve_model_dir(config().get("models_dir"))}', flush=True)
    print('[启动] 正在检查程序更新并初始化服务…', flush=True)
    try:
        if updater.apply_on_start():
            os.execv(sys.executable, [sys.executable, '-m', 'hi3d_workbench', *sys.argv[1:]])
    except Exception as exc:
        if updater.recovery_required():
            raise RuntimeError('Interrupted update could not be restored; see data/updates/backups') from exc
        print(f'Code update was not installed; using existing program: {exc}', flush=True)
    lifecycle.shutdown.clear()
    lifecycle.managed=True
    print(f'[启动] 访问地址：http://127.0.0.1:{args.port}/ （服务就绪后可用）', flush=True)
    threading.Thread(target=console_control.report_when_ready,args=(args.port,args.open),daemon=True).start()
    import uvicorn
    server=uvicorn.Server(uvicorn.Config("hi3d_workbench.server:app",host="127.0.0.1",port=args.port,access_log=False))
    def stop_requested():
        lifecycle.shutdown.wait()
        server.should_exit=True
    threading.Thread(target=stop_requested,daemon=True).start()
    try:server.run()
    finally:
        lifecycle.managed=False
        lifecycle.shutdown.set()
    return 0

if __name__=="__main__":raise SystemExit(main())
