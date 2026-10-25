using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
[assembly: System.Reflection.AssemblyTitle("Hi3D Workbench Launcher")]
[assembly: System.Reflection.AssemblyProduct("Hi3D T8")]
[assembly: System.Reflection.AssemblyCopyright("T8star")]

// Assign the suspended child before it can create inference subprocesses.
internal static class WorkbenchLauncher
{
    private static IntPtr job;
    private static uint childId;
    private static int stopping;
    private static Timer stopTimer;
    private static readonly Handler handler = OnControl;
    private delegate bool Handler(uint signal);
    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimits
    {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinWorkingSet, MaxWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters { public ulong A, B, C, D, E, F; }
    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimits
    {
        public BasicLimits Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct StartupInfo
    {
        public int Size;
        public string Reserved, Desktop, Title;
        public uint X, Y, XSize, YSize, XChars, YChars, Fill, Flags;
        public short ShowWindow, ReservedSize;
        public IntPtr ReservedBytes, Input, Output, Error;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInfo { public IntPtr Process, Thread; public uint Id, ThreadId; }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr handle, int kind, ref ExtendedLimits info, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr handle, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr handle, uint code);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(string application, StringBuilder command, IntPtr pa, IntPtr ta,
        bool inherit, uint flags, IntPtr environment, string directory, ref StartupInfo startup, out ProcessInfo process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll")]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll")]
    private static extern bool GetExitCodeProcess(IntPtr process, out uint code);
    [DllImport("kernel32.dll")]
    private static extern bool TerminateProcess(IntPtr process, uint code);
    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll")]
    private static extern bool SetConsoleCtrlHandler(Handler callback, bool add);
    [DllImport("kernel32.dll")]
    private static extern bool GenerateConsoleCtrlEvent(uint signal, uint group);
    private static void Check(bool ok) { if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error()); }

    private static bool OnControl(uint signal)
    {
        if (signal == 0 || signal == 1)
        {
            if (Interlocked.Exchange(ref stopping, 1) == 0 && childId != 0)
            {
                Console.WriteLine("\n正在关闭工作台并释放推理进程，请稍候…");
                GenerateConsoleCtrlEvent(1, childId);
                stopTimer = new Timer(delegate { if (job != IntPtr.Zero) TerminateJobObject(job, 1); }, null, 10000, Timeout.Infinite);
            }
            return true;
        }
        // Window close/logoff: Windows terminates all members on handle close.
        IntPtr owned = Interlocked.Exchange(ref job, IntPtr.Zero);
        if (owned != IntPtr.Zero) CloseHandle(owned);
        return false;
    }

    private static string Quote(string value)
    {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value)
        {
            if (c == '\\') { slashes++; continue; }
            result.Append('\\', c == '"' ? slashes * 2 + 1 : slashes);
            result.Append(c); slashes = 0;
        }
        result.Append('\\', slashes * 2);
        return result.Append('"').ToString();
    }

    private static int Main(string[] args)
    {
        int exitCode = 1;
        IntPtr processHandle = IntPtr.Zero;
        try
        {
            Console.OutputEncoding = new UTF8Encoding(false);
            try { Console.Title = "Hi3D 工作台 · T8star · 关闭本窗口即可停止"; } catch (IOException) { }
            string root = AppDomain.CurrentDomain.BaseDirectory;
            Directory.SetCurrentDirectory(root);
            Console.WriteLine("Hi3D 本地 3D 工作台 · By T8star\n");
            Console.WriteLine("此窗口是工作台服务，请保持打开。\n关闭方式：Ctrl+C / 关闭此窗口 / 网页中的“关闭工作台”。\n只关闭浏览器页面不会停止服务。\n");
            string python = null;
            foreach (string name in new[] { @"runtime\python.exe", @".appenv\Scripts\python.exe", @".venv\Scripts\python.exe" })
                if (File.Exists(Path.Combine(root, name))) { python = Path.Combine(root, name); break; }
            if (python == null) throw new FileNotFoundException("没有找到 Python 环境。请使用完整版整合包，或先按 README 安装运行环境。");
            foreach (string name in new[] { "PYTHONHOME", "PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "HI3D_WORKBENCH_ROOT" })
                Environment.SetEnvironmentVariable(name, null);
            Environment.SetEnvironmentVariable("PYTHONIOENCODING", "utf-8");
            Environment.SetEnvironmentVariable("TRITON_CACHE_DIR", Path.Combine(root, @"data\kernel-cache"));
            string compiler = Path.Combine(root, @"runtime\Lib\site-packages\triton\runtime\tcc\tcc.exe");
            if (File.Exists(compiler)) Environment.SetEnvironmentVariable("CC", compiler);
            job = CreateJobObject(IntPtr.Zero, null);
            Check(job != IntPtr.Zero);
            var limits = new ExtendedLimits(); limits.Basic.Flags = 0x2000; // KILL_ON_JOB_CLOSE
            Check(SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(limits)));
            Check(SetConsoleCtrlHandler(handler, true));
            var command = new StringBuilder(Quote(python) + " -u -m hi3d_workbench --open");
            foreach (string arg in args) command.Append(" ").Append(Quote(arg));
            var startup = new StartupInfo(); startup.Size = Marshal.SizeOf(startup);
            ProcessInfo child;
            Check(CreateProcess(python, command, IntPtr.Zero, IntPtr.Zero, false,
                0x00000004 | 0x00000200, IntPtr.Zero, root, ref startup, out child));
            processHandle = child.Process;
            try
            {
                if (!AssignProcessToJobObject(job, child.Process))
                {
                    int error = Marshal.GetLastWin32Error();
                    TerminateProcess(child.Process, 1);
                    throw new Win32Exception(error, "无法建立进程退出保护，工作台未启动。");
                }
                childId = child.Id;
                if (ResumeThread(child.Thread) == uint.MaxValue) throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            finally { CloseHandle(child.Thread); }
            WaitForSingleObject(child.Process, uint.MaxValue);
            uint result; Check(GetExitCodeProcess(child.Process, out result));
            exitCode = stopping != 0 ? 0 : unchecked((int)result);
        }
        catch (Exception error) { Console.Error.WriteLine("启动失败：" + error.Message); }
        finally
        {
            if (stopTimer != null) stopTimer.Dispose();
            IntPtr owned = Interlocked.Exchange(ref job, IntPtr.Zero);
            if (owned != IntPtr.Zero) CloseHandle(owned);
            if (processHandle != IntPtr.Zero) CloseHandle(processHandle);
            childId = 0;
            SetConsoleCtrlHandler(handler, false);
        }
        Console.WriteLine("\n本次启动已结束，本窗口所属的服务和子进程已释放。\n若上方提示“已在运行”，请从原窗口或网页关闭已有服务。\n按回车关闭此窗口。");
        if (stopping == 0 && !Console.IsInputRedirected) Console.ReadLine();
        return exitCode;
    }
}
