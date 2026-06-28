"""Diagnosis layer: turn detections into plain-language fault reports.

This is deterministic rule-based mapping, NOT more ML -- the same idea a real
ECU uses when it maps a detected condition to a Diagnostic Trouble Code (DTC)
with a human-readable description. It consumes ONLY detector output: at runtime
there is no ground truth, only symptoms. (The evaluation layer is the mirror
image -- it uses labels and never runs in production.)

DTC note: real OBD-II codes use a letter prefix for the domain -- P powertrain,
C chassis, B body, U network/communication. The codes below follow that
*convention* (U... for comms faults, P... for signal faults) but are
project-internal symbols, not standard OEM codes, which are vehicle-specific.
"""
from dataclasses import dataclass
from typing import Dict, List

from .detectors import Detection
from .model import Vehicle


# How much each ECU matters if it misbehaves. For an EV, brakes and steering are
# safety-critical and the HV battery (BMS) sits close behind.
ECU_CRITICALITY: Dict[str, str] = {
    "Brake": "critical",
    "Steering": "critical",
    "BMS": "high",
    "Engine": "high",
    "TurnSignal": "low",
}

# fault_class -> (code, category, headline template, likely cause)
SIGNATURES: Dict[str, tuple] = {
    "SILENCE": ("U-COMM-LOSS", "Network / communication", "{ecu} stopped communicating",
                "ECU reset, wiring/connector fault, or the node went bus-off"),
    "DELAY":   ("U-COMM-DEGRADED", "Network / communication", "{ecu} communication unstable",
                "message delay or jitter -- bus overload, partial failure, or scheduling issue"),
    "BURST":   ("U-BUS-FLOOD", "Network / communication", "abnormal traffic from {ecu}",
                "babbling node or traffic injection -- can starve higher-priority messages"),
    "VALUE":   ("P-SIG-RANGE", "Signal plausibility", "{ecu} reporting an implausible value",
                "sensor fault, stuck value, or an out-of-range reading"),
    "UNKNOWN": ("U-UNKNOWN", "Unclassified", "anomaly on {ecu}", "unclassified detector output"),
}


@dataclass
class Diagnosis:
    ecu: str
    can_id: int
    code: str
    category: str
    headline: str
    likely_cause: str
    severity: str          # Low / Medium / High
    t_start: float
    t_end: float
    duration: float
    evidence: str          # which detector + reason produced this (traceability)


def _classify(d: Detection) -> str:
    r = d.reason
    if "silence" in r:                       # silence dominates even if delay also fired
        return "SILENCE"
    if d.detector == "value" or "value:" in r:
        return "VALUE"
    if "burst" in r or "count_high" in r:
        return "BURST"
    if "delay" in r or "gap" in r:
        return "DELAY"
    return "UNKNOWN"


def _severity(fault_class: str, criticality: str, duration: float) -> str:
    sustained = duration >= 2.0
    if fault_class == "SILENCE":
        return "Low" if criticality == "low" else "High"
    if fault_class == "BURST":
        return "High"                        # a babbling node degrades the whole bus
    if fault_class == "DELAY":
        if criticality == "critical":
            return "High" if sustained else "Medium"
        return "Low" if criticality == "low" else "Medium"
    if fault_class == "VALUE":
        if criticality == "critical":
            return "High"
        return "Low" if criticality == "low" else "Medium"
    return "Medium"


def diagnose(detections: List[Detection], vehicle: Vehicle) -> List[Diagnosis]:
    names = {m.can_id: m.name for m in vehicle.messages}
    out: List[Diagnosis] = []
    for d in detections:
        ecu = names.get(d.can_id, f"0x{d.can_id:03X}")
        fault_class = _classify(d)
        code, category, headline_t, cause = SIGNATURES[fault_class]
        crit = ECU_CRITICALITY.get(ecu, "medium")
        duration = d.t_end - d.t_start

        headline = headline_t.format(ecu=ecu)
        if fault_class == "VALUE" and "value:" in d.reason:   # name the offending signal
            sig = d.reason.split("value:")[1].split(" ")[0]
            headline = f"{ecu} reporting an implausible {sig}"

        out.append(Diagnosis(
            ecu=ecu, can_id=d.can_id, code=code, category=category, headline=headline,
            likely_cause=cause, severity=_severity(fault_class, crit, duration),
            t_start=round(d.t_start, 3), t_end=round(d.t_end, 3),
            duration=round(duration, 3), evidence=f"{d.detector} detector -- {d.reason}",
        ))

    order = {"High": 0, "Medium": 1, "Low": 2}
    out.sort(key=lambda x: (order.get(x.severity, 3), x.t_start))   # most serious first
    return out


def format_diagnoses(diags: List[Diagnosis]) -> str:
    if not diags:
        return "No faults diagnosed -- bus nominal."
    L = ["=== Diagnosis (from detector output; no ground truth used) ===", ""]
    for d in diags:
        L.append(f"[{d.severity.upper()}] {d.headline}  ({d.code})")
        L.append(f"    window:   {d.t_start:.1f}-{d.t_end:.1f} s  ({d.duration:.1f} s)")
        L.append(f"    category: {d.category}")
        L.append(f"    likely:   {d.likely_cause}")
        L.append(f"    evidence: {d.evidence}")
        L.append("")
    return "\n".join(L).rstrip()
