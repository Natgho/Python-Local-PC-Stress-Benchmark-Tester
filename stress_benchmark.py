#!/usr/bin/env python3
"""
stress_benchmark.py — simple CPU & RAM stress/benchmark tool (pure Python)

Features
- CPU burn across N processes (default: all logical cores)
- RAM allocation to a target (GB or % of total) with optional churn
- Fixed-duration runs with periodic progress reports
- Pure Python, no extra deps, cross‑platform (Linux/macOS/Windows)
- Safe defaults; use CLI flags to go "full throttle"

USAGE EXAMPLES
--------------
# 30s full CPU on all cores, 8 GB RAM allocation with churn
python3 stress_benchmark.py --seconds 30 --cpu all --mem-gb 8 --mem-churn

# 2 minutes, CPU on all cores, fill 70% of RAM (gradual), report every 2s
python3 stress_benchmark.py --seconds 120 --cpu all --mem-percent 70 --report-interval 2

# Gentle sanity check (defaults): ~50% cores, 10% RAM, 20s
python3 stress_benchmark.py

DISCLAIMER
----------
This tool can make your system sluggish or unresponsive if you excessively stress it.
Use on your own machine, save all work first, and prefer running within a controlled environment.

OWNER
-----
Sezer BOZKIR <admin@sezerbozkir.com>

RELEASE DATES
----------------
17-10-2025 - initial release
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
import math
from typing import List

# ---------- Utilities ----------

def total_memory_bytes() -> int:
    """Best-effort total physical memory in bytes without external deps."""
    try:
        if sys.platform == "darwin":  # macOS
            # sysctl hw.memsize
            import subprocess, shlex
            out = subprocess.check_output(shlex.split("sysctl -n hw.memsize")).strip()
            return int(out)
        elif os.name == "posix":  # Linux/Unix
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
        elif os.name == "nt":  # Windows
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
    except Exception:
        pass
    # Fallback: assume 8GB if unknown (conservative for calculations)
    return 8 * (1024**3)

def human_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b/1024:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"

# ---------- Workloads ----------

def cpu_burn(stop: mp.Event, intensity: int = 1) -> None:
    """
    Tight math loop to burn a core. 'intensity' roughly scales loop work per tick.
    """
    # Localize to avoid global lookups
    sin = math.sin
    sqrt = math.sqrt
    x = 0.0001
    k = 0
    # Busy loop
    while not stop.is_set():
        # Unrolled numeric work
        for _ in range(1000 * intensity):
            x = sin(x) * sqrt(abs(x) + 1.0) + 1.0000001
            k += 1
        # Very short sleep to be slightly scheduler-friendly (optional)
        # Removing sleep may provide marginally higher CPU but worse responsiveness
        # time.sleep(0)

def touch_chunk(buf: bytearray) -> None:
    """Write to each page to ensure memory is committed and stays 'hot'."""
    if not buf:
        return
    page = 4096
    ln = len(buf)
    for i in range(0, ln, page):
        buf[i] = (buf[i] + 1) & 0xFF

def memory_balloons(target_bytes: int, chunk_mb: int = 64, churn: bool = False, stop: mp.Event | None = None) -> List[bytearray]:
    """
    Allocate memory up to target_bytes in chunk_mb increments.
    If churn=True, periodically touch pages to keep the allocator busy.
    Returns the list of chunks to keep them referenced.
    """
    chunks: List[bytearray] = []
    allocated = 0
    chunk_bytes = chunk_mb * 1024 * 1024
    try:
        while allocated < target_bytes and (stop is None or not stop.is_set()):
            remain = target_bytes - allocated
            this = min(chunk_bytes, remain)
            buf = bytearray(this)
            # Touch to force physical commitment
            touch_chunk(buf)
            chunks.append(buf)
            allocated += this
            # Gradual ramp to avoid sudden OOM
            time.sleep(0.05)
        # Optional churn loop
        if churn:
            while stop is None or not stop.is_set():
                for buf in chunks:
                    touch_chunk(buf)
                # Small pause so we don't starve CPU (RAM churn is significant)
                time.sleep(0.1)
    except MemoryError:
        # We hit an OOM barrier; keep what we allocated and continue churn (if requested)
        pass
    return chunks

# ---------- Runner ----------

def run_benchmark(seconds: int, cpu_workers: int, mem_bytes: int, mem_chunk_mb: int, mem_churn: bool, report_interval: int):
    # Graceful stop event
    stop = mp.Event()

    # Spawn CPU workers
    procs: List[mp.Process] = []
    for _ in range(cpu_workers):
        p = mp.Process(target=cpu_burn, args=(stop, 1), daemon=True)
        p.start()
        procs.append(p)

    # Spawn memory ballooner in separate process so churn doesn't block
    mem_holder = mp.Manager().list()  # placeholder to keep reference across process
    def mem_target(stop_evt, size_bytes, chunk_mb, churn):
        _ = memory_balloons(size_bytes, chunk_mb, churn, stop_evt)
        # Keep alive until stop
        while not stop_evt.is_set():
            time.sleep(0.2)

    mp_ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context()
    mem_proc = None
    if mem_bytes > 0:
        mem_proc = mp_ctx.Process(target=mem_target, args=(stop, mem_bytes, mem_chunk_mb, mem_churn), daemon=True)
        mem_proc.start()

    # Reporting loop
    start = time.time()
    next_report = start
    try:
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= seconds:
                break
            if now >= next_report:
                # We don't have psutil; keep it simple
                print(f"[{elapsed:6.1f}s] CPU workers: {cpu_workers} | Target RAM: {human_bytes(mem_bytes)}")
                next_report = now + report_interval
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Interrupted by user, stopping...")
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=2)
        if mem_proc is not None:
            mem_proc.join(timeout=2)

def parse_cpu_workers(arg: str | None) -> int:
    cores = os.cpu_count() or 1
    if arg is None:
        # Default: half the cores, rounded up, as a safe baseline
        return max(1, (cores + 1) // 2)
    if arg.strip().lower() in ("all", "max"):
        return cores
    try:
        n = int(arg)
        if n < 1:
            raise ValueError
        return min(n, cores)
    except Exception:
        raise argparse.ArgumentTypeError("Invalid --cpu value. Use 'all' or a positive integer.")

def clamp_mem_target(total_bytes: int, mem_gb: float | None, mem_percent: float | None) -> int:
    if mem_gb is not None and mem_percent is not None:
        raise ValueError("Specify only one of --mem-gb or --mem-percent.")
    if mem_gb is None and mem_percent is None:
        # Default: 10% of total, but not more than 4GB
        return int(min(total_bytes * 0.10, 4 * (1024**3)))
    if mem_gb is not None:
        target = int(mem_gb * (1024**3))
    else:
        if not (0 < mem_percent <= 95):
            raise ValueError("--mem-percent must be within (0, 95].")
        target = int(total_bytes * (mem_percent / 100.0))
    # Keep a safety headroom of ~5% to reduce OOM risk
    headroom = int(total_bytes * 0.05)
    return max(0, min(target, total_bytes - headroom))

def main():
    parser = argparse.ArgumentParser(description="CPU & RAM stress/benchmark tool (pure Python)")
    parser.add_argument("--seconds", "-s", type=int, default=20, help="Total duration to run (default: 20)")
    parser.add_argument("--cpu", type=str, default=None, help="CPU workers: 'all' or integer count (default: ~half cores)")
    parser.add_argument("--mem-gb", type=float, default=None, help="Target RAM to allocate in GB (mutually exclusive with --mem-percent)")
    parser.add_argument("--mem-percent", type=float, default=None, help="Target RAM to allocate as %% of total (mutually exclusive with --mem-gb)")
    parser.add_argument("--mem-chunk-mb", type=int, default=64, help="Allocation chunk size in MB (default: 64)")
    parser.add_argument("--mem-churn", action="store_true", help="Continuously touch allocated memory pages")
    parser.add_argument("--report-interval", type=int, default=1, help="Seconds between progress reports (default: 1)")

    args = parser.parse_args()

    total_bytes = total_memory_bytes()
    try:
        mem_bytes = clamp_mem_target(total_bytes, args.mem_gb, args.mem_percent)
    except ValueError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        sys.exit(2)

    cpu_workers = parse_cpu_workers(args.cpu)

    print("=== stress_benchmark.py ===")
    print(f"Detected logical cores : {os.cpu_count()}")
    print(f"Total physical memory  : {human_bytes(total_bytes)}")
    print(f"CPU workers            : {cpu_workers}")
    print(f"Target RAM             : {human_bytes(mem_bytes)} (chunk {args.mem_chunk_mb} MB, churn={args.mem_churn})")
    print(f"Duration               : {args.seconds} s")
    print("-----------------------------")

    run_benchmark(
        seconds=args.seconds,
        cpu_workers=cpu_workers,
        mem_bytes=mem_bytes,
        mem_chunk_mb=args.mem_chunk_mb,
        mem_churn=args.mem_churn,
        report_interval=args.report_interval,
    )

if __name__ == "__main__":
    # Make sure multiprocessing works reliably on macOS/Windows
    mp = __import__("multiprocessing")
    if sys.platform == "darwin":
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    main()
