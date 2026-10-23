import argparse
import time
import json
from .models import download_all,STATE,verify_all,status

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','verify','status']);args=p.parse_args()
    if args.command=='download':
        if not download_all():
            print('已有模型下载进程运行，请等待它完成。')
            raise SystemExit(2)
        last=''
        while STATE['state'] not in {'complete','failed'}:
            message=STATE.get('message','')
            if message!=last:print(message,flush=True);last=message
            time.sleep(2)
        print(json.dumps(STATE,ensure_ascii=False,indent=2))
        if STATE['state']=='failed':raise SystemExit(1)
    else:
        result=verify_all() if args.command=='verify' else status()
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.command=='verify' and not result['ok']:raise SystemExit(1)

if __name__=='__main__':main()
