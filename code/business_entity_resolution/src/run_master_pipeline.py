"""
run_master_pipeline.py
======================
Autonomous supervisor for Steps 2-4:
  Step 2: Candidate Blocking on Test Set
  Step 3: Fast Vectorized Test Inference
  Step 4: Official Submission Validation
  Bonus : Generate submission package zip

Includes active background health & battery monitoring:
- Checks battery level and AC wall power status every 30s.
- Low-battery safeguard if power cut occurs (< 20% on battery).
- C: and D: drive disk space monitor.
- Writes real-time telemetry to output/pipeline_status.json and output/pipeline_run.log.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import psutil

ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT_DIR / "output"
LOG_FILE = OUTPUT_DIR / "pipeline_run.log"
STATUS_FILE = OUTPUT_DIR / "pipeline_status.json"

stop_monitor = threading.Event()


def log_msg(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def monitor_loop():
    """Background thread monitoring battery, RAM, and disk space."""
    while not stop_monitor.is_set():
        try:
            battery = psutil.sensors_battery()
            ram = psutil.virtual_memory()
            disk_c = psutil.disk_usage("C:")
            disk_d = psutil.disk_usage("D:")

            batt_pct = battery.percent if battery else 100.0
            batt_plugged = battery.power_plugged if battery else True

            status = {
                "timestamp": datetime.now().isoformat(),
                "battery": {
                    "percent": batt_pct,
                    "plugged_in": batt_plugged,
                },
                "ram": {
                    "used_gb": round(ram.used / (1024**3), 2),
                    "total_gb": round(ram.total / (1024**3), 2),
                    "percent": ram.percent,
                },
                "disk_c_free_gb": round(disk_c.free / (1024**3), 2),
                "disk_d_free_gb": round(disk_d.free / (1024**3), 2),
            }

            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)

            # Safety check: Power cut detection & battery low
            if not batt_plugged and batt_pct <= 18.0:
                log_msg(f"[CRITICAL ALERT] Battery at {batt_pct}% and running on battery! Power cut detected!")

        except Exception as e:
            pass

        time.sleep(30)


def run_command_logged(cmd: list[str], stage_name: str) -> bool:
    log_msg(f"=== Starting: {stage_name} ===")
    log_msg(f"Command: {' '.join(cmd)}")
    start_t = time.time()

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        line_clean = line.rstrip()
        if line_clean:
            print(f"[{stage_name}] {line_clean}", flush=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{stage_name}] {line_clean}\n")

    process.stdout.close()
    return_code = process.wait()
    duration = time.time() - start_t

    if return_code == 0:
        log_msg(f"=== PASSED: {stage_name} (Elapsed: {duration:.1f}s) ===\n")
        return True
    else:
        log_msg(f"=== FAILED: {stage_name} (Exit code: {return_code}, Elapsed: {duration:.1f}s) ===\n")
        return False


def build_submission_zip():
    log_msg("=== Building Final Submission ZIP Archive ===")
    zip_path = OUTPUT_DIR / "final_submission.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. output/ files
            m_file = OUTPUT_DIR / "matching_results.tsv"
            c_file = OUTPUT_DIR / "candidate_pairs.tsv"
            if m_file.exists():
                zf.write(m_file, arcname="output/matching_results.tsv")
            if c_file.exists():
                zf.write(c_file, arcname="output/candidate_pairs.tsv")

            # 2. code/business_entity_resolution/src
            src_dir = ROOT_DIR / "code" / "business_entity_resolution" / "src"
            if src_dir.exists():
                for root, _, files in os.walk(src_dir):
                    for file in files:
                        if not file.endswith((".pyc", ".pyo")):
                            fp = Path(root) / file
                            arcname = "code/business_entity_resolution/src/" + fp.relative_to(src_dir).as_posix()
                            zf.write(fp, arcname=arcname)

            # 3. Documentation_template.md
            doc_file = ROOT_DIR / "Documentation_template.md"
            if doc_file.exists():
                zf.write(doc_file, arcname="Documentation_template.md")

        zip_mb = round(zip_path.stat().st_size / (1024**2), 2)
        log_msg(f"Submission ZIP created successfully: {zip_path} ({zip_mb} MB)")
    except Exception as e:
        log_msg(f"Failed to create ZIP: {e}")


def main():
    log_msg("Starting Master Pipeline Supervisor...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    py_exe = sys.executable

    # -------------------------------------------------------------
    # Step 2: Test Blocking
    # -------------------------------------------------------------
    cand_file = OUTPUT_DIR / "candidate_pairs.tsv"
    if cand_file.exists() and cand_file.stat().st_size > 100_000:
        log_msg(f"Step 2: Candidate pairs already exist at {cand_file} ({cand_file.stat().st_size:,} bytes). Skipping.")
        step2_ok = True
    else:
        step2_cmd = [
            py_exe, "-X", "utf8",
            "code/business_entity_resolution/src/blocking.py",
            "--split", "test",
            "--output-dir", "output",
            "--artifacts-dir", "output/artifacts_test",
        ]
        step2_ok = run_command_logged(step2_cmd, "Step 2: Test Blocking")

    if not step2_ok:
        log_msg("Pipeline aborted due to Step 2 failure.")
        stop_monitor.set()
        return

    # -------------------------------------------------------------
    # Step 3: Test Inference
    # -------------------------------------------------------------
    step3_cmd = [
        py_exe, "-X", "utf8",
        "code/business_entity_resolution/src/inference.py",
        "--candidate-file", "output/candidate_pairs.tsv",
        "--model-dir", "output/models",
        "--output-dir", "output",
    ]
    step3_ok = run_command_logged(step3_cmd, "Step 3: Test Inference")

    if not step3_ok:
        log_msg("Pipeline aborted due to Step 3 failure.")
        stop_monitor.set()
        return

    # -------------------------------------------------------------
    # Step 4: Submission Validation
    # -------------------------------------------------------------
    step4_cmd = [
        py_exe,
        "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test",
    ]
    step4_ok = run_command_logged(step4_cmd, "Step 4: Submission Validation")

    if step4_ok:
        log_msg("Validation PASSED! Ready for submission.")
        build_submission_zip()
    else:
        log_msg("Validation completed with issues — please review logs.")

    log_msg("=== Master Pipeline Finished ===")
    stop_monitor.set()


if __name__ == "__main__":
    main()
