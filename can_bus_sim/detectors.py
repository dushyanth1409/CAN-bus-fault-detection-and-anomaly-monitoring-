"""Interpretable anomaly detectors.

Two detectors, on purpose:

  TimingDetector  -> when frames arrive.   Catches silence, delay, burst.
  ValueDetector   -> what frames contain.  Catches out-of-range signals.

Both learn their normal behaviour from the clean trace (no hard-coded cycle
times or signal limits) and emit Detection objects with a `fire_time` -- the
earliest moment the evidence is complete -- so detection latency is measured
honestly rather than retroactively.

Why interpretable models: the project's whole point is to *explain* a fault in
plain terms. A detector you can trace to "Engine went silent for >1 cycle" or
"pack_voltage left its learned range" gives the diagnosis layer something to say.
A black-box reconstruction error does not.
"""
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .model import Frame, Vehicle


@dataclass
class Detection:
    can_id: int
    t_start: float       # start of the anomalous span
    t_end: float         # end of the anomalous span
    fire_time: float     # earliest time the detection is knowable (for latency)
    detector: str
    reason: str


def group_by_id(frames: List[Frame]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = defaultdict(list)
    for f in frames:
        out[f.can_id].append(f.timestamp)
    for k in out:
        out[k].sort()
    return out


def window_counts(ts: List[float], W: float, T: float) -> List[int]:
    n = int(T // W)  # full windows only -> no partial-trailing-window edge effect
    counts = [0] * max(n, 1)
    for t in ts:
        i = int(t // W)
        if 0 <= i < n:
            counts[i] += 1
    return counts


def merge(dets: List[Detection], gap_tol: float = 0.5) -> List[Detection]:
    """Merge same-(id, detector) detections whose spans overlap or are within gap_tol."""
    groups: Dict = defaultdict(list)
    for d in dets:
        groups[(d.can_id, d.detector)].append(d)
    out: List[Detection] = []
    for ds in groups.values():
        ds.sort(key=lambda d: d.t_start)
        cur = None
        for d in ds:
            if cur and d.t_start <= cur.t_end + gap_tol:
                cur.t_end = max(cur.t_end, d.t_end)
                cur.fire_time = min(cur.fire_time, d.fire_time)
                cur.reason = "+".join(sorted(set(cur.reason.split("+")) | set(d.reason.split("+"))))
            else:
                if cur:
                    out.append(cur)
                cur = Detection(d.can_id, d.t_start, d.t_end, d.fire_time, d.detector, d.reason)
        if cur:
            out.append(cur)
    out.sort(key=lambda d: (d.t_start, d.can_id))
    return out


class TimingDetector:
    """Per-ID watchdog (deadline) + per-ID per-window frame-count band."""

    def __init__(self, window_s: float = 0.25):
        self.W = window_s
        self.count_med: Dict[int, float] = {}
        self.gap_max: Dict[int, float] = {}

    def fit(self, frames: List[Frame]) -> None:
        T = max(f.timestamp for f in frames)
        for cid, ts in group_by_id(frames).items():
            iats = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            self.gap_max[cid] = (max(iats) * 1.8 + 0.002) if iats else math.inf
            counts = window_counts(ts, self.W, T)
            self.count_med[cid] = statistics.median(counts) if counts else 0

    def predict(self, frames: List[Frame]) -> List[Detection]:
        T = max(f.timestamp for f in frames)
        by_id = group_by_id(frames)
        dets: List[Detection] = []

        # Count band: catches silence (count collapses) and burst (count explodes).
        for cid, ts in by_id.items():
            if cid not in self.count_med:
                continue
            med = self.count_med[cid]
            lo, hi = 0.5 * med, 2 * med + 5
            for i, c in enumerate(window_counts(ts, self.W, T)):
                ws, we = i * self.W, (i + 1) * self.W
                if c < lo:
                    dets.append(Detection(cid, ws, we, we, "timing", "count_low (silence)"))
                elif c > hi:
                    dets.append(Detection(cid, ws, we, we, "timing", "count_high (burst)"))

        # Watchdog: a deadline breach fires at last_seen + gap_max, not at resume.
        for cid, ts in by_id.items():
            g = self.gap_max.get(cid)
            if g is None or g == math.inf:
                continue
            for j in range(len(ts) - 1):
                if ts[j + 1] - ts[j] > g:
                    fire = ts[j] + g
                    dets.append(Detection(cid, fire, ts[j + 1], fire, "timing", "gap/delay"))
            if ts and (T - ts[-1]) > g:  # silence that runs to the end of the trace
                fire = ts[-1] + g
                dets.append(Detection(cid, fire, T, fire, "timing", "gap/silence"))

        return merge(dets)


class ValueDetector:
    """Per-(ID, signal) plausibility band learned from clean data."""

    def __init__(self, vehicle: Vehicle, margin_frac: float = 0.05):
        self.by_id = vehicle.by_id()
        self.margin_frac = margin_frac
        self.bands: Dict = {}

    def fit(self, frames: List[Frame]) -> None:
        acc: Dict = defaultdict(list)
        for f in frames:
            msg = self.by_id.get(f.can_id)
            if not msg:
                continue
            for name, val in msg.decode(f.data).items():
                acc[(f.can_id, name)].append(val)
        for key, vals in acc.items():
            lo, hi = min(vals), max(vals)
            m = self.margin_frac * (hi - lo) + 1e-6
            self.bands[key] = (lo - m, hi + m)

    def predict(self, frames: List[Frame]) -> List[Detection]:
        dets: List[Detection] = []
        for f in frames:
            msg = self.by_id.get(f.can_id)
            if not msg:
                continue
            for name, val in msg.decode(f.data).items():
                band = self.bands.get((f.can_id, name))
                if not band:
                    continue
                lo, hi = band
                if val < lo or val > hi:
                    dets.append(Detection(f.can_id, f.timestamp, f.timestamp,
                                          f.timestamp, "value", f"value:{name} out of range"))
        return merge(dets)
