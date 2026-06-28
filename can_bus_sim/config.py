"""The simulated vehicle: four periodic ECUs + one event-triggered message.

Cycle times are chosen to be realistic in spirit (safety/powertrain fast,
body/energy slow) and CAN IDs follow the convention that a lower ID = higher
priority. The signal dynamics are stylised but deterministic functions of time
so a given seed always reproduces the same trace.

The event-triggered TurnSignal is included on purpose: it forces the detector
(Phase 2) to handle non-periodic traffic instead of assuming everything is
cyclic. A timing detector that flags it as "missing" every few seconds is
wrong, and you want that failure mode visible from day one.
"""
import math

from .model import Message, Signal, Vehicle
from .faults import FaultEvent


def build_vehicle() -> Vehicle:
    # --- Engine: 0x0C0, 10 ms (fast, high priority) ---
    engine = Message(
        can_id=0x0C0, name="Engine", cycle_ms=10.0, dlc=8,
        signals=[
            Signal("rpm", 0, 2, scale=0.25, offset=0.0, minimum=0, maximum=8000, unit="rpm",
                   dynamics=lambda t: 800 + 600 * math.sin(2 * math.pi * t / 30)
                                      + 350 * max(0.0, math.sin(2 * math.pi * t / 7))),
            Signal("coolant_temp", 2, 1, scale=1.0, offset=-40.0, minimum=-40, maximum=140, unit="C",
                   dynamics=lambda t: 20 + 70 * (1 - math.exp(-t / 45))),
        ],
    )

    # --- Brake: 0x080, 10 ms (highest priority, safety) ---
    def brake_pressure(t: float) -> float:
        return 60.0 if (t % 15.0) < 1.0 else 0.0  # ~1 s press every 15 s

    brake = Message(
        can_id=0x080, name="Brake", cycle_ms=10.0, dlc=8,
        signals=[
            Signal("brake_pressure", 0, 1, scale=1.0, offset=0.0, minimum=0, maximum=200, unit="bar",
                   dynamics=brake_pressure),
            Signal("brake_active", 1, 1, scale=1.0, offset=0.0, minimum=0, maximum=1, unit="bool",
                   dynamics=lambda t: 1.0 if brake_pressure(t) > 0 else 0.0),
        ],
    )

    # --- Steering: 0x100, 20 ms ---
    steering = Message(
        can_id=0x100, name="Steering", cycle_ms=20.0, dlc=8,
        signals=[
            Signal("steer_angle", 0, 2, scale=0.1, offset=-1000.0, minimum=-540, maximum=540, unit="deg",
                   dynamics=lambda t: 30 * math.sin(2 * math.pi * t / 12)),
            Signal("steer_torque", 2, 1, scale=0.05, offset=-6.0, minimum=-6, maximum=6, unit="Nm",
                   dynamics=lambda t: 1.5 * math.sin(2 * math.pi * t / 12 + 0.3)),
        ],
    )

    # --- Battery management: 0x200, 100 ms (slow, lower priority) ---
    bms = Message(
        can_id=0x200, name="BMS", cycle_ms=100.0, dlc=8,
        signals=[
            Signal("pack_voltage", 0, 2, scale=0.1, offset=0.0, minimum=300, maximum=420, unit="V",
                   dynamics=lambda t: 384 + 12 * math.sin(2 * math.pi * t / 90)),
            Signal("soc", 2, 1, scale=0.5, offset=0.0, minimum=0, maximum=100, unit="%",
                   dynamics=lambda t: max(0.0, 78 - 0.02 * t)),
            Signal("pack_current", 3, 1, scale=1.0, offset=-50.0, minimum=-50, maximum=200, unit="A",
                   dynamics=lambda t: 20 + 40 * max(0.0, math.sin(2 * math.pi * t / 25))),
        ],
    )

    # --- Turn signal: 0x350, event-triggered (~every 7 s on average) ---
    turn = Message(
        can_id=0x350, name="TurnSignal", dlc=1, event_triggered=True, mean_interval_s=7.0,
        signals=[
            Signal("direction", 0, 1, scale=1.0, offset=0.0, minimum=0, maximum=2, unit="enum",
                   dynamics=lambda t: float(1 + (int(t) % 2))),  # 1=left, 2=right
        ],
    )

    return Vehicle(messages=[brake, engine, steering, bms, turn])


def default_campaign(duration_s: float):
    """A demo fault campaign placed proportionally so it fits any duration.

    One fault per category in your taxonomy, so Phase 2 has something of each
    kind to detect:
      - missing   -> silence (Engine goes quiet)
      - timing    -> delay   (Brake messages arrive late and jittery)
      - value     -> value   (BMS pack_voltage drifts out of range)
      - traffic   -> burst   (Steering floods the bus)
    """
    d = duration_s

    def win(start_frac, length_s):
        start = start_frac * d
        return start, min(d, start + length_s)

    s1, e1 = win(0.30, 6.0)
    s2, e2 = win(0.50, 8.0)
    s3, e3 = win(0.68, 6.0)
    s4, e4 = win(0.85, 2.0)

    return [
        FaultEvent(0x0C0, "silence", s1, e1, {}),
        FaultEvent(0x080, "delay", s2, e2, {"extra_ms": 60, "jitter_ms": 25}),
        FaultEvent(0x200, "value", s3, e3, {"signal": "pack_voltage", "mode": "offset", "amount": 60.0}),
        FaultEvent(0x100, "burst", s4, e4, {"period_ms": 1.0}),
    ]
