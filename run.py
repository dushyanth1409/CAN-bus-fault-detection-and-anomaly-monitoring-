"""CLI: generate a clean trace, a faulted trace, and the ground-truth labels.

Usage:
    python -m can_bus_sim.run --duration 60 --seed 0 --out-dir out

Outputs (in --out-dir):
    clean_trace.csv    fault-free bus traffic (train your "normal" model on this)
    faulted_trace.csv  same traffic with the campaign applied (test on this)
    labels.json        the injected faults = ground truth for scoring
"""
import argparse
import json
import os
from typing import List

from .config import build_vehicle, default_campaign
from .faults import FaultEvent
from .generator import generate
from .faults import apply_campaign
from .model import Frame, Vehicle


def write_trace(frames: List[Frame], path: str) -> None:
    with open(path, "w") as fh:
        fh.write("timestamp,can_id,name,dlc,data_hex\n")
        for f in frames:
            fh.write(f.to_csv_row() + "\n")


def write_labels(campaign: List[FaultEvent], vehicle: Vehicle, path: str) -> None:
    by_id = vehicle.by_id()
    labels = [{
        "target_id": f"0x{ev.target_id:03X}",
        "target_name": by_id[ev.target_id].name,
        "fault_type": ev.fault_type,
        "t_start": round(ev.t_start, 6),
        "t_end": round(ev.t_end, 6),
        "params": ev.params,
    } for ev in campaign]
    with open(path, "w") as fh:
        json.dump(labels, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="CAN data generator + fault injection harness")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of traffic")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible runs)")
    ap.add_argument("--out-dir", default="out", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vehicle = build_vehicle()

    clean = generate(vehicle, args.duration, seed=args.seed)
    write_trace(clean, os.path.join(args.out_dir, "clean_trace.csv"))

    campaign = default_campaign(args.duration)
    faulted = apply_campaign(clean, vehicle, campaign, seed=args.seed)
    write_trace(faulted, os.path.join(args.out_dir, "faulted_trace.csv"))
    write_labels(campaign, vehicle, os.path.join(args.out_dir, "labels.json"))

    print(f"duration={args.duration}s seed={args.seed}")
    print(f"clean frames:   {len(clean)}")
    print(f"faulted frames: {len(faulted)}  (delta {len(faulted) - len(clean):+d})")
    print(f"faults injected: {len(campaign)} -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
