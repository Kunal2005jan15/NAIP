# =============================================================
# NAIP Project - Step 00: Full Live-Data Refresh Orchestrator
# =============================================================
# Runs the entire LIVE pipeline (weather pull -> predictions ->
# alerts -> sanity gate) in the correct order, with logging, so
# this can be triggered by Windows Task Scheduler instead of
# manually running 8 scripts by hand.
#
# Order matters - each step depends on the previous one's output:
#   22 -> fetch raw current weather (NASA POWER)
#   26 -> fix -999 fill-value placeholders
#   25 -> build current-season feature inputs (Rabi window etc.)
#   35 -> fetch live NDVI (Earth Engine - needs prior auth token)
#   27 -> generate current predictions (+ SHAP) from the model
#   28 -> District Watch snapshot + diff against previous
#   32 -> Kharif early-warning risk classification
#   33 -> automated sanity gate - the LAST line of defense before
#         the dashboard reads any of this
#
# If a step fails, the chain STOPS (downstream steps would just
# fail on stale/missing input anyway) - except the final sanity
# gate, whose failures are reported clearly but don't crash the
# orchestrator, since it's informational at that point.
# =============================================================

import subprocess
import sys
import os
from datetime import datetime

PIPELINE_STEPS = [
    'src/22_fetch_current_weather.py',
    'src/26_fix_fill_values.py',
    'src/25_current_conditions.py',
    'src/35_fetch_ndvi_current.py',
    'src/27_generate_current_predictions.py',
    'src/28_district_watch.py',
    'src/32_kharif_early_warning.py',
    'src/33_pipeline_sanity_checks.py',
]

os.makedirs('logs', exist_ok=True)
log_path = f"logs/refresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(line):
    print(line)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log(f"{'='*60}")
log(f"NAIP FULL REFRESH - started {datetime.now().isoformat()}")
log(f"{'='*60}")

overall_ok = True

# Force UTF-8 for subprocess output. On Windows, output captured
# through a pipe (subprocess.run with capture_output=True) falls
# back to the system codepage (often cp1252) instead of UTF-8,
# which crashes on any Unicode character a script prints (checkmarks,
# arrows, etc.) - even though the SAME script runs fine printing
# straight to a real terminal. This isn't a bug in the pipeline
# scripts themselves; it's specific to capturing their output.
subprocess_env = os.environ.copy()
subprocess_env['PYTHONIOENCODING'] = 'utf-8'

for i, step in enumerate(PIPELINE_STEPS, 1):
    is_last = (i == len(PIPELINE_STEPS))
    log(f"\n[{i}/{len(PIPELINE_STEPS)}] Running {step} ...")

    result = subprocess.run(
        [sys.executable, step],
        capture_output=True, text=True, encoding='utf-8',
        env=subprocess_env
    )
    log(result.stdout)
    if result.stderr:
        log(f"--- stderr ---\n{result.stderr}")

    if result.returncode != 0:
        if is_last:
            # The sanity gate exits non-zero on FAILED checks by
            # design (see script 33) - that's a signal to read, not
            # an orchestrator crash. Report clearly and stop here.
            log(f"\n[GATE FAILED] {step} reported failed checks (exit code {result.returncode}).")
            log("Refresh data was generated, but the sanity gate found a problem.")
            log("DO NOT trust the dashboard/predictions until this is resolved -")
            log(f"see the full gate output above, or rerun: python {step}")
            overall_ok = False
        else:
            log(f"\n[STOPPED] {step} failed (exit code {result.returncode}). "
                f"Downstream steps depend on this succeeding - not continuing.")
            overall_ok = False
        break
    else:
        log(f"[OK] {step} completed.")

log(f"\n{'='*60}")
if overall_ok:
    log(f"REFRESH COMPLETE - all {len(PIPELINE_STEPS)} steps succeeded, sanity gate passed.")
else:
    log("REFRESH INCOMPLETE OR FLAGGED - see above for details.")
log(f"Log saved to {log_path}")
log(f"{'='*60}")

sys.exit(0 if overall_ok else 1)