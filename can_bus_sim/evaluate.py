"""Score detections against ground-truth labels.

Time-series anomalies are ranges, not points, so scoring is overlap-based:
a detection is a true positive if it overlaps (within a small tolerance) a
labelled fault on the same ID. Reporting:

  recall     fraction of labelled faults caught by >=1 detection
  precision  fraction of detection regions that hit a real fault
  latency    per fault, time from its onset to the first detection's fire_time

A small match tolerance is allowed because a fault's effect bleeds a little past
its labelled boundary (e.g. the last delayed frame lands just after t_end).
"""
from typing import Dict, List, Tuple

from .detectors import Detection


def _overlap(a0: float, a1: float, b0: float, b1: float, tol: float) -> bool:
    return a0 <= b1 + tol and b0 - tol <= a1


def evaluate(detections: List[Detection], labels: List[Dict],
             match_tol: float = 0.3) -> Tuple[List[Dict], Dict]:
    per_fault: List[Dict] = []
    for lab in labels:
        lid = lab["target_id_int"]
        hits = [d for d in detections
                if d.can_id == lid and _overlap(d.t_start, d.t_end, lab["t_start"], lab["t_end"], match_tol)]
        detected = len(hits) > 0
        latency = max(0.0, min(d.fire_time for d in hits) - lab["t_start"]) if detected else None
        per_fault.append({
            "id": lab["target_id"], "name": lab["target_name"], "type": lab["fault_type"],
            "t_start": lab["t_start"], "t_end": lab["t_end"], "detected": detected,
            "by": sorted({d.detector for d in hits}), "latency": latency,
        })

    tp = sum(1 for d in detections
             if any(d.can_id == lab["target_id_int"]
                    and _overlap(d.t_start, d.t_end, lab["t_start"], lab["t_end"], match_tol)
                    for lab in labels))
    summary = {
        "recall": sum(r["detected"] for r in per_fault) / len(labels) if labels else 1.0,
        "precision": tp / len(detections) if detections else 1.0,
        "tp_regions": tp, "fp_regions": len(detections) - tp, "n_detections": len(detections),
        "match_tol": match_tol,
    }
    return per_fault, summary


def format_report(per_fault: List[Dict], summary: Dict, clean_fp: int, window_s: float) -> str:
    L = []
    L.append("=== CAN fault detection report (faulted trace) ===")
    L.append(f"detectors: timing (window={window_s}s, watchdog + count band), value (learned range)")
    L.append(f"false positives on the clean trace: {clean_fp}")
    L.append("")
    L.append(f"{'ID':<7}{'name':<11}{'type':<9}{'window':<15}{'detected':<10}{'by':<10}{'latency':<9}")
    for r in per_fault:
        win = f"{r['t_start']:.1f}-{r['t_end']:.1f}s"
        lat = f"{r['latency']:.3f}s" if r["latency"] is not None else "-"
        L.append(f"{r['id']:<7}{r['name']:<11}{r['type']:<9}{win:<15}"
                 f"{('YES' if r['detected'] else 'NO'):<10}{(','.join(r['by']) or '-'):<10}{lat:<9}")
    L.append("")
    L.append(f"recall:    {summary['recall'] * 100:.0f}%  "
             f"({sum(1 for r in per_fault if r['detected'])}/{len(per_fault)} faults)")
    L.append(f"precision: {summary['precision'] * 100:.0f}%  "
             f"({summary['tp_regions']}/{summary['n_detections']} regions, "
             f"{summary['fp_regions']} false positive, match tol {summary['match_tol']}s)")
    L.append("")
    L.append("Read this honestly: these faults were injected well above detector")
    L.append("thresholds, so high scores are expected. The experiment that earns its")
    L.append("keep is sweeping fault magnitude (and adding bus noise) until precision")
    L.append("and recall start to drop -- that curve is the actual result.")
    return "\n".join(L)
