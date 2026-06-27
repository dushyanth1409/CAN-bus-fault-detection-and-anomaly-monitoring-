"""Virtual-clock CAN traffic generator (normal behaviour).

No time.sleep, no real-time coupling. Each message keeps a "next send time";
we always advance to the earliest one, emit a frame, and reschedule. This is a
discrete-event simulation: an hour of bus traffic is generated in a fraction of
a second, and the output is fully reproducible for a given seed.

Normal timing is not perfectly periodic -- real ECUs have small jitter -- so we
add a little Gaussian noise to each interval. That jitter is what makes the
"normal" model non-trivial: the Phase 2 timing detector has to learn the
spread, not just the mean.
"""
import heapq
import random
from typing import List

from .model import Frame, Vehicle


def generate(vehicle: Vehicle, duration_s: float, seed: int = 0,
             jitter_frac: float = 0.02) -> List[Frame]:
    rng = random.Random(seed)
    heap = []          # (next_time, tiebreak_seq, message)
    seq = 0

    for msg in vehicle.messages:
        if msg.event_triggered:
            first = rng.expovariate(1.0 / msg.mean_interval_s)
        else:
            first = rng.uniform(0.0, msg.cycle_ms / 1000.0)  # random phase so not all fire at t=0
        heapq.heappush(heap, (first, seq, msg))
        seq += 1

    frames: List[Frame] = []
    while heap:
        t, _, msg = heapq.heappop(heap)
        if t > duration_s:
            continue
        frames.append(Frame(t, msg.can_id, msg.name, msg.dlc, msg.encode(t)))

        if msg.event_triggered:
            nxt = t + rng.expovariate(1.0 / msg.mean_interval_s)
        else:
            cycle = msg.cycle_ms / 1000.0
            nxt = t + cycle + rng.gauss(0.0, jitter_frac * cycle)
        if nxt <= duration_s:
            heapq.heappush(heap, (nxt, seq, msg))
            seq += 1

    frames.sort(key=lambda f: f.timestamp)  # jitter can reorder neighbours; keep trace monotonic
    return frames
