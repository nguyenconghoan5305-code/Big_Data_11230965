"""Sample resident memory of a benchmark process and its descendants."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys


if sys.platform == "win32":
    class _ProcessEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _kernel.Process32FirstW.restype = wintypes.BOOL
    _kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    _kernel.Process32NextW.restype = wintypes.BOOL
    _kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel.OpenProcess.restype = wintypes.HANDLE
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
    _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def _windows_tree(root_pid: int) -> set[int]:
    handle = _kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    parents: dict[int, int] = {}
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        found = _kernel.Process32FirstW(handle, ctypes.byref(entry))
        while found:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            found = _kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        _kernel.CloseHandle(handle)
    tree = {root_pid}
    while True:
        expanded = tree | {pid for pid, parent in parents.items() if parent in tree}
        if expanded == tree:
            return tree
        tree = expanded


def _windows_rss(pid: int) -> int | None:
    handle = _kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return None
    try:
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    finally:
        _kernel.CloseHandle(handle)


def _linux_tree(root_pid: int) -> set[int]:
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            parents[int(path.parent.name)] = int(raw.rsplit(") ", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    tree = {root_pid}
    while True:
        expanded = tree | {pid for pid, parent in parents.items() if parent in tree}
        if expanded == tree:
            return tree
        tree = expanded


def _linux_rss(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def sample_tree_rss(root_pid: int) -> tuple[int, int, int]:
    """Return summed RSS bytes, visible process count, unreadable process count."""
    if sys.platform == "win32":
        pids, reader = _windows_tree(root_pid), _windows_rss
    elif sys.platform.startswith("linux"):
        pids, reader = _linux_tree(root_pid), _linux_rss
    else:
        raise RuntimeError(f"Unsupported process-tree sampler: {sys.platform}")
    values = [reader(pid) for pid in pids]
    return sum(value for value in values if value is not None), len(pids), sum(value is None for value in values)
