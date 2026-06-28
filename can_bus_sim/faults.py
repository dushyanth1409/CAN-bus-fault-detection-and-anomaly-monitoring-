"""Fault injection layer.

A FaultEvent is a scheduled, fully-specified corruption: which ID, what kind,
when it starts and ends, and any parameters. The list of events IS the ground
truth -- run.py writes the same list out as labels.json. There is no hidden or
probabilistic corruption that isn't recorded, which is exactly what lets you
score a detector later (a detection at time t on ID x is a true positive iff it
falls inside a matching event window).

Faults are applied to an already-generated clean trace rather than during
generation. That keeps "normal behaviour" and "what went wrong" in separate
files you can diff, and lets you reuse one clean trace to train on.
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List

from .model import Frame, Vehicle


@dataclass
class FaultEvent:
    target_id: int
    fault_type: str          # 'silence' | 'intermittent_drop' | 'delay' | 'value' | 'burst'
    t_start: float
    t_end: float
    params: Dict = field(default_factory=dict)

    def covers(self, frame: Frame) -> bool:
        return frame.can_id == self.target_id and self.t_start <= frame.timestamp <= self.t_end


def apply_campaign(frames: List[Frame], vehicle: Vehicle,
                   campaign: List[FaultEvent], seed: int = 0) -> List[Frame]:
    rng = random.Random(seed + 1)      # distinct stream from the generator
    by_id = vehicle.by_id()
    out = list(frames)

    for ev in campaign:
        if ev.fault_type == "silence":
            # Node goes quiet: drop every frame of this ID inside the window.
            out = [f for f in out if not ev.covers(f)]

        elif ev.fault_type == "intermittent_drop":
            # Flaky link: drop each in-window frame with some probability.
            p = ev.params.get("probability", 0.3)
            out = [f for f in out if not (ev.covers(f) and rng.random() < p)]

        elif ev.fault_type == "delay":
            # Late + jittery arrivals: push timestamps later within the window.
            extra = ev.params.get("extra_ms", 50) / 1000.0
            jit = ev.params.get("jitter_ms", 20) / 1000.0
            out = [
                Frame(f.timestamp + extra + abs(rng.gauss(0.0, jit)),
                      f.can_id, f.name, f.dlc, f.data) if ev.covers(f) else f
                for f in out
            ]

        elif ev.fault_type == "value":
            # Signal fault: decode, perturb one signal, re-encode.
            msg = by_id[ev.target_id]
            sig = ev.params["signal"]
            mode = ev.params.get("mode", "offset")
            new = []
            for f in out:
                if ev.covers(f):
                    values = msg.decode(f.data)
                    if mode == "stuck":
                        values[sig] = ev.params["value"]
                    elif mode == "offset":
                        values[sig] += ev.params["amount"]
                    elif mode == "noise":
                        values[sig] += rng.uniform(-ev.params["amount"], ev.params["amount"])
                    new.append(Frame(f.timestamp, f.can_id, f.name, f.dlc,
                                     msg.encode_from_values(values)))
                else:
                    new.append(f)
            out = new

        elif ev.fault_type == "burst":
            # Babbling node / flood: extra frames at a much higher rate.
            msg = by_id[ev.target_id]
            period = ev.params.get("period_ms", 1.0) / 1000.0
            t = ev.t_start
            extra = []
            while t <= ev.t_end:
                extra.append(Frame(t, msg.can_id, msg.name, msg.dlc, msg.encode(t)))
                t += period
            out = out + extra

        else:
            raise ValueError(f"unknown fault_type: {ev.fault_type}")

    out.sort(key=lambda f: f.timestamp)
    return out
