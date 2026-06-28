"""CLI: run the detectors, diagnose the result, and (if labels exist) score it.

Usage:
    python -m can_bus_sim.detect --out-dir out
    python -m can_bus_sim.detect --no-eval        # runtime mode: diagnosis only

Two outputs with different purposes:
  Diagnosis   consumes only detector output -> the operator/runtime view.
  Evaluation  consumes ground-truth labels  -> the test scorecard (never in prod).
"""
import argparse
import json
import os
from dataclasses import asdict

from .config import build_vehicle
from .detectors import TimingDetector, ValueDetector
from .diagnosis import diagnose, format_diagnoses
from .evaluate import evaluate, format_report
from .traceio import load_labels, load_trace


def run_detectors(timing, value, frames):
    return timing.predict(frames) + value.predict(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CAN fault detectors, diagnose, and score")
    ap.add_argument("--clean", default="out/clean_trace.csv")
    ap.add_argument("--faulted", default="out/faulted_trace.csv")
    ap.add_argument("--labels", default="out/labels.json")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--window", type=float, default=0.25)
    ap.add_argument("--no-eval", action="store_true",
                    help="skip scoring (runtime mode: no ground truth available)")
    args = ap.parse_args()

    vehicle = build_vehicle()
    clean = load_trace(args.clean)
    faulted = load_trace(args.faulted)

    timing = TimingDetector(window_s=args.window)
    value = ValueDetector(vehicle)
    timing.fit(clean)
    value.fit(clean)

    clean_fp = len(run_detectors(timing, value, clean))     # should be ~0
    detections = run_detectors(timing, value, faulted)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "detections.json"), "w") as fh:
        json.dump([{
            "can_id": f"0x{d.can_id:03X}", "t_start": round(d.t_start, 4),
            "t_end": round(d.t_end, 4), "fire_time": round(d.fire_time, 4),
            "detector": d.detector, "reason": d.reason,
        } for d in detections], fh, indent=2)

    # --- Diagnosis: operator view, built from detections only (no labels) ---
    diags = diagnose(detections, vehicle)
    print(format_diagnoses(diags))
    print()
    with open(os.path.join(args.out_dir, "diagnoses.json"), "w") as fh:
        json.dump([{**asdict(d), "can_id": f"0x{d.can_id:03X}"} for d in diags], fh, indent=2)

    # --- Evaluation: test scorecard, needs ground-truth labels ---
    if not args.no_eval and os.path.exists(args.labels):
        labels = load_labels(args.labels)
        per_fault, summary = evaluate(detections, labels)
        report = format_report(per_fault, summary, clean_fp, args.window)
        print(report)
        with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
            fh.write(report + "\n")


if __name__ == "__main__":
    main()
