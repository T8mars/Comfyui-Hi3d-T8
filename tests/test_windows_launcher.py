"""Real Windows Job Object checks with a disposable worker, no GPU or models."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows console launcher')
ROOT = Path(__file__).resolve().parents[1]


def wait_until(check, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():return
        time.sleep(.1)
    raise AssertionError('Timed out waiting for launcher process lifecycle')


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / 'bundle 中文 with spaces'
    root.mkdir()
    subprocess.run(['powershell', '-NoProfile', '-File', str(ROOT/'scripts/Build-Launcher.ps1'),
                    '-OutputDirectory', str(root)], check=True, capture_output=True)
    runtime = root/'runtime'; runtime.mkdir()
    stub = root/'FakeWorker.cs'
    stub.write_text('''using System; using System.IO; using System.Diagnostics; using System.Threading;
class FakeWorker {
 static void Main(string[] args) {
  Console.WriteLine("WORKER_STDOUT_VISIBLE"); Console.Out.Flush();
  Console.Error.WriteLine("WORKER_STDERR_VISIBLE"); Console.Error.Flush();
  if (args.Length > 0 && args[0] == "leaf") { Thread.Sleep(Timeout.Infinite); return; }
  Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e) {
   File.WriteAllText("stopped.txt", e.SpecialKey.ToString()); Environment.Exit(0);
  };
  var start = new ProcessStartInfo(Process.GetCurrentProcess().MainModule.FileName, "leaf");
  start.UseShellExecute = false; start.CreateNoWindow = true;
  var leaf = Process.Start(start);
  File.WriteAllText("worker.pid", Process.GetCurrentProcess().Id.ToString());
  File.WriteAllText("leaf.pid", leaf.Id.ToString());
  Thread.Sleep(Timeout.Infinite);
 }
}''', encoding='utf-8')
    compiler = Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(compiler), '/nologo', '/target:exe', '/platform:x64',
                    '/out:'+str(runtime/'python.exe'), str(stub)], check=True, capture_output=True)
    return root


def test_launcher_forwards_worker_stdout_and_stderr(launcher):
    # Exercise the actual CreateProcess boundary; a standalone Python/HTTP test
    # cannot detect missing inherited handles in the native launcher.
    info = subprocess.STARTUPINFO(); info.dwFlags |= subprocess.STARTF_USESHOWWINDOW; info.wShowWindow = 0
    stdout_path, stderr_path = launcher/'stdout.log', launcher/'stderr.log'
    with stdout_path.open('wb') as out, stderr_path.open('wb') as err:
        process = subprocess.Popen([str(launcher/'Hi3D-Workbench.exe')], cwd=launcher,
                                   stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=info)
        try:
            wait_until(lambda: (launcher/'leaf.pid').exists())
            wait_until(lambda: b'WORKER_STDOUT_VISIBLE' in stdout_path.read_bytes()
                       and b'WORKER_STDERR_VISIBLE' in stderr_path.read_bytes(), timeout=3)
        finally:
            process.kill()
            process.wait(timeout=10)
        for name in ('worker.pid','leaf.pid'):
            wait_until(lambda name=name: not psutil.pid_exists(int((launcher/name).read_text())))


def start(root):
    info = subprocess.STARTUPINFO(); info.dwFlags |= subprocess.STARTF_USESHOWWINDOW; info.wShowWindow = 0
    process = subprocess.Popen([str(root/'Hi3D-Workbench.exe')], cwd=root,
                               creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=info)
    wait_until(lambda: (root/'leaf.pid').exists())
    return process, [int((root/name).read_text()) for name in ('worker.pid','leaf.pid')]


@pytest.mark.parametrize('action', ['kill_launcher', 'close_window', 'ctrl_c', 'worker_exit'])
def test_launcher_owns_all_descendants(launcher, action):
    process, descendants = start(launcher)
    try:
        assert all(psutil.pid_exists(pid) for pid in descendants)
        if action == 'kill_launcher':process.kill()
        else:
            helper = '''import ctypes,time,sys
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.FreeConsole()
assert k.AttachConsole(int(sys.argv[1]))
k.SetConsoleCtrlHandler(None,True)
if sys.argv[2]=='ctrl_c': assert k.GenerateConsoleCtrlEvent(0,0)
elif sys.argv[2]=='worker_exit': assert k.GenerateConsoleCtrlEvent(1,int(sys.argv[3]))
else:
 k.GetConsoleWindow.restype=ctypes.c_void_p
 window=k.GetConsoleWindow()
 k.FreeConsole()
 assert ctypes.windll.user32.PostMessageW(ctypes.c_void_p(window),0x0010,0,0)
time.sleep(.5)
k.FreeConsole()
'''
            subprocess.run([sys.executable, '-c', helper, str(process.pid), action, str(descendants[0])], check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        wait_until(lambda: not any(psutil.pid_exists(pid) for pid in descendants))
        if action == 'worker_exit':assert process.poll() is None  # Keeps logs visible after server exit.
        else:wait_until(lambda: process.poll() is not None)
        if action == 'ctrl_c':assert (launcher/'stopped.txt').read_text() == 'ControlBreak'
    finally:
        if process.poll() is None:process.kill()
        process.wait(timeout=10)
        for pid in descendants:
            try:psutil.Process(pid).kill()
            except psutil.NoSuchProcess:pass
