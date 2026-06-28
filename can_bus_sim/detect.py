"""CLI: run the detectors and score them against ground truth.

Usage:
    python -m can_bus_sim.detect \
        --clean out/clean_trace.csv --faulted out/faulted_trace.csv \
        --labels out/labels.json --out-dir out
"""
import argparse
import json
import os

from .config import build_vehicle
from .detectors import TimingDetector, ValueDetector
from .evaluate import evaluate, format_report
from .traceio import load_labels, load_trace


def run_detectors(timing, value, frames):
    return timing.predict(frames) + value.predict(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CAN fault detectors and score them")
    ap.add_argument("--clean", default="out/clean_trace.csv")
    ap.add_argument("--faulted", default="out/faulted_trace.csv")
    ap.add_argument("--labels", default="out/labels.json")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--window", type=float, default=0.25)
    args = ap.parse_args()

    vehicle = build_vehicle()
    clean = load_trace(args.clean)
    faulted = load_trace(args.faulted)
    labels = load_labels(args.labels)

    timing = TimingDetector(window_s=args.window)
    value = ValueDetector(vehicle)
    timing.fit(clean)
    value.fit(clean)

    clean_fp = len(run_detectors(timing, value, clean))     # should be ~0
    detections = run_detectors(timing, value, faulted)

    per_fault, summary = evaluate(detections, labels)
    report = format_report(per_fault, summary, clean_fp, args.window)
    print(report)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "detections.json"), "w") as fh:
        json.dump([{
            "can_id": f"0x{d.can_id:03X}", "t_start": round(d.t_start, 4),
            "t_end": round(d.t_end, 4), "fire_time": round(d.fire_time, 4),
            "detector": d.detector, "reason": d.reason,
        } for d in detections], fh, indent=2)


if __name__ == "__main__":
    main()
