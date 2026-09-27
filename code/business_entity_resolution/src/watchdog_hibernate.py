"""
watchdog_hibernate.py
=====================
Monitors two conditions:
1. Battery charge drops <= 40% (e.g. during a power cut) -> hibernates immediately.
2. Master pipeline completes (final_submission.zip created) -> hibernates system safely.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "output"
LOG_FILE = OUTPUT_DIR / "watchdog.log"
ZIP_FILE = OUTPUT_DIR / "final_submission.zip"
PIPELINE_LOG = OUTPUT_DIR / "pipeline_run.log"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [WATCHDOG] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def hibernate():
    log("Executing system hibernation: shutdown /h ...")
    time.sleep(3)
    os.system("shutdown /h")


def main():
    log("Hibernate watchdog started.")
    log("Rules: Hibernate if battery <= 40% OR when pipeline finishes.")

    while True:
        try:
            # 1. Check Battery
            b = psutil.sensors_battery()
            if b is not None:
                # If battery drops to <= 40% on battery power
                if not b.power_plugged and b.percent <= 40.0:
                    log(f"ALERT: Battery at {b.percent}% on battery power (<= 40%). Hibernating now to preserve battery!")
                    hibernate()
                    break

            # 2. Check Pipeline Completion
            if ZIP_FILE.exists() and ZIP_FILE.stat().st_size > 100_000:
                # Double-check that pipeline_run.log has recorded completion
                if PIPELINE_LOG.exists():
                    try:
                        content = PIPELINE_LOG.read_text(encoding="utf-8", errors="ignore")
                        if "=== Master Pipeline Finished ===" in content:
                            log("Master pipeline completed successfully and final_submission.zip is ready!")
                            log("Waiting 15 seconds to ensure all file writes are flushed, then hibernating...")
                            time.sleep(15)
                            hibernate()
                            break
                    except Exception:
                        pass

        except Exception as e:
            log(f"Watchdog exception: {e}")

        time.sleep(15)


if __name__ == "__main__":
    main()
