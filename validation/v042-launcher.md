# v0.4.2 Windows launcher validation

2026-10-25. No model, inference algorithm or ComfyUI canvas changes.

- Windows x64 EXE compiled from `scripts/launcher/WorkbenchLauncher.cs` using the Windows .NET Framework 4 compiler. Python and model weights are separate.
- Disposable native worker plus detached grandchild tested in a directory containing spaces and Chinese characters. Ctrl+C reaches the server process group as Ctrl+Break. Closing the actual console window, killing the launcher and normal console cancellation all remove its descendants through a Windows Job Object.
- Real Python workbench subprocess tested with no GPU task: HTTP readiness, all seven social/resource URLs, duplicate launch reuse, rejection of a different bundle directory, and authenticated CLI shutdown.
- Shutdown API rejects missing session tokens, cross-site requests and servers outside the managed console entry point.
- Code update archive checked by the installed v0.4.1 updater: no new root executable or stop-script entry is required by its protected-path allowlist. Launcher compilation and stop-script creation are separate from the code overlay. Models and runtimes are excluded from public code assets.
- Browser automation was unavailable on this host for this run; visual browser interaction is not claimed. New controls and resource links were checked through HTTP and application tests. No new GPU inference was required or claimed for this launcher change.
