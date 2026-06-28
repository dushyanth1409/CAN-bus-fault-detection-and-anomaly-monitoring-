"""Read/write helpers for traces and labels.

Single source of truth for the on-disk format so the generator and the detector
can never disagree about the schema.
"""
import json
from typing import Dict, List

from .model import Frame


def write_trace(frames: List[Frame], path: str) -> None:
    with open(path, "w") as fh:
        fh.write("timestamp,can_id,name,dlc,data_hex\n")
        for f in frames:
            fh.write(f.to_csv_row() + "\n")


def load_trace(path: str) -> List[Frame]:
    frames: List[Frame] = []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            ts, cid, name, dlc, data_hex = line.rstrip("\n").split(",")
            frames.append(Frame(float(ts), int(cid, 16), name, int(dlc), bytes.fromhex(data_hex)))
    return frames


def load_labels(path: str) -> List[Dict]:
    with open(path) as fh:
        labels = json.load(fh)
    for lab in labels:
        lab["target_id_int"] = int(lab["target_id"], 16)  # convenience for matching
    return labels
