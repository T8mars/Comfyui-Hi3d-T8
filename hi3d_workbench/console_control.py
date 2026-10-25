"""Control only the local server belonging to this exact bundle directory."""
import json
import socket
import time
import urllib.request
import webbrowser
from . import lifecycle


def request(port, path='/api/session', token=None):
    headers = {'X-Hi3D-Token': token} if token else {}
    value = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                                   data=b'{}' if token else None, headers=headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(value, timeout=2) as response:
        return json.load(response)


def running(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.3):return True
    except OSError:return False


def existing(port, stop=False, open_browser=False):
    if not running(port):
        if stop:
            print('本目录工作台没有运行，无需关闭。', flush=True)
            return 0
        return None
    try:
        session = request(port)
        if session.get('instance_id') != lifecycle.instance_id():
            raise ValueError('该端口已被其他目录或旧版程序占用，请关闭原启动窗口或更改本目录端口。')
        if stop:
            if not session.get('can_shutdown'):raise ValueError('该服务需要从原启动窗口关闭。')
            request(port, '/api/shutdown', token=session['token'])
            print('已请求关闭本目录工作台，正在释放任务和 GPU 进程。', flush=True)
        else:
            print(f'本目录工作台已在 http://127.0.0.1:{port}/ 运行，没有重复启动。\n'
                  '请使用原启动窗口、网页“关闭工作台”或 Stop-Workbench.cmd 停止服务。', flush=True)
            if open_browser:webbrowser.open(f'http://127.0.0.1:{port}/')
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'无法操作本地服务：{exc}', flush=True)
        return 1


def report_when_ready(port, open_browser=False, deadline=90, report_interval=5):
    """Report actual HTTP readiness even when browser opening is disabled.

    Long initial imports must not leave an apparently stalled console. The
    deadline changes the message, not the running server or its initialization.
    """
    started = time.monotonic()
    next_report = started + report_interval
    while not lifecycle.shutdown.is_set():
        try:
            if request(port).get('instance_id') == lifecycle.instance_id():
                address = f'http://127.0.0.1:{port}/'
                print(f'\n[就绪] 工作台已启动：{address}', flush=True)
                if open_browser:
                    try:opened = webbrowser.open(address)
                    except (OSError, webbrowser.Error):opened = False
                    if opened:
                        print('[浏览器] 已发送打开请求；页面未出现时请手动打开上面的地址。', flush=True)
                    else:
                        print('[浏览器] 未能自动打开，请手动打开上面的地址。', flush=True)
                print('[运行] 等待任务；任务进度将在此窗口与网页中显示。', flush=True)
                return
        except (OSError, ValueError):pass
        now = time.monotonic()
        if now >= next_report:
            elapsed = int(now - started)
            detail = '初始化耗时较长，请查看上方日志；服务尚未就绪' if elapsed >= deadline else '正在初始化服务，尚未就绪'
            print(f'[启动] {detail}（已等待 {elapsed} 秒）', flush=True)
            next_report = now + report_interval
        lifecycle.shutdown.wait(.5)


def open_when_ready(port, deadline=90):
    report_when_ready(port, open_browser=True, deadline=deadline)
