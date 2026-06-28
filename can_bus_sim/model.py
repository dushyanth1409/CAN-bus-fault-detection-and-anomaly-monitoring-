"""Bus model: signals, messages, frames, and encode/decode.

This is the "what lives on the bus" layer. It is deliberately kept separate
from the generator (which decides *when* frames are sent) and the fault layer
(which decides *how* the stream gets corrupted). That separation is the whole
reason the harness is testable: you can produce one clean trace and one
faulted trace from the same model and compare them against known labels.

Simplification vs. a real DBC: signals here are byte-aligned (each occupies a
whole number of bytes, little-endian). Real DBC signals sit at arbitrary bit
offsets. encode()/decode() are the only place that would change to lift that
restriction, so the simplification is contained and easy to defend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Signal:
    name: str
    start_byte: int                 # byte offset within the data field
    length_bytes: int               # 1 or 2 in this model
    scale: float = 1.0              # physical = raw * scale + offset
    offset: float = 0.0
    minimum: float = 0.0           # plausible range (used downstream, not enforced here)
    maximum: float = 0.0
    unit: str = ""
    dynamics: Optional[Callable[[float], float]] = None  # physical value as a function of t (seconds)

    def encode(self, physical_value: float) -> bytes:
        raw = round((physical_value - self.offset) / self.scale)
        max_raw = (1 << (8 * self.length_bytes)) - 1
        raw = max(0, min(max_raw, raw))  # clamp to the field width
        return raw.to_bytes(self.length_bytes, "little")

    def decode(self, data: bytes) -> float:
        chunk = data[self.start_byte:self.start_byte + self.length_bytes]
        raw = int.from_bytes(chunk, "little")
        return raw * self.scale + self.offset


@dataclass
class Message:
    can_id: int
    name: str
    signals: List[Signal] = field(default_factory=list)
    dlc: int = 8
    cycle_ms: Optional[float] = None        # nominal period; None for event-triggered
    event_triggered: bool = False
    mean_interval_s: float = 5.0            # only used when event_triggered

    def values_at(self, t: float) -> Dict[str, float]:
        return {s.name: (s.dynamics(t) if s.dynamics else s.minimum) for s in self.signals}

    def encode_from_values(self, values: Dict[str, float]) -> bytes:
        data = bytearray(self.dlc)
        for s in self.signals:
            val = values.get(s.name, s.minimum)
            data[s.start_byte:s.start_byte + s.length_bytes] = s.encode(val)
        return bytes(data)

    def encode(self, t: float) -> bytes:
        return self.encode_from_values(self.values_at(t))

    def decode(self, data: bytes) -> Dict[str, float]:
        return {s.name: s.decode(data) for s in self.signals}


@dataclass
class Frame:
    timestamp: float
    can_id: int
    name: str
    dlc: int
    data: bytes

    def to_csv_row(self) -> str:
        return f"{self.timestamp:.6f},0x{self.can_id:03X},{self.name},{self.dlc},{self.data.hex()}"


@dataclass
class Vehicle:
    """A bundle of messages = the bus definition (a tiny stand-in for a DBC)."""
    messages: List[Message] = field(default_factory=list)

    def by_id(self) -> Dict[int, Message]:
        return {m.can_id: m for m in self.messages}
