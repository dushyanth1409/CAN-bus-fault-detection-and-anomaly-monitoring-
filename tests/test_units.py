"""
Unit tests for the CAN bus fault-detection internals.

Where the integration suite (test_pipeline.py) exercises the CLI end-to-end,
this suite tests the individual pieces directly against their real signatures:

  model.py      encode/decode round-trip + field clamping + CSV schema
  detectors.py  window counting, merge, watchdog fire-time, value margin band
  diagnosis.py  the _classify and _severity rules, branch by branch
  evaluate.py   overlap matching, recall/precision, latency clamping

Run from the repo root:  python -m pytest -v
"""
import math

import pytest

from can_bus_sim.model import Signal, Message, Frame, Vehicle
from can_bus_sim.detectors import (
    Detection, TimingDetector, ValueDetector,
    group_by_id, window_counts, merge,
)
from can_bus_sim.diagnosis import _classify, _severity, diagnose
from can_bus_sim.evaluate import _overlap, evaluate


# --------------------------------------------------------------------------- #
# model.py                                                                     #
# --------------------------------------------------------------------------- #
class TestSignalCodec:
    def test_encode_decode_round_trip_with_scale(self):
        # scale=0.1 lets us represent sub-integer physical values exactly.
        sig = Signal("v", start_byte=0, length_bytes=2, scale=0.1)
        data = sig.encode(385.0)
        assert sig.decode(data) == pytest.approx(385.0)

    def test_round_trip_with_offset(self):
        sig = Signal("temp", start_byte=0, length_bytes=1, scale=1.0, offset=-40.0)
        assert sig.decode(sig.encode(0.0)) == pytest.approx(0.0)

    def test_encode_clamps_above_field_width(self):
        # 1-byte field -> max raw 255. 300 must clamp, not overflow/wrap.
        sig = Signal("x", start_byte=0, length_bytes=1, scale=1.0)
        assert sig.decode(sig.encode(300.0)) == pytest.approx(255.0)

    def test_encode_clamps_below_zero(self):
        sig = Signal("x", start_byte=0, length_bytes=1, scale=1.0)
        assert sig.decode(sig.encode(-5.0)) == pytest.approx(0.0)


class TestMessageCodec:
    def test_multi_signal_round_trip_respects_byte_layout(self):
        a = Signal("a", start_byte=0, length_bytes=1, scale=1.0)
        b = Signal("b", start_byte=1, length_bytes=2, scale=0.5, offset=100.0)
        msg = Message(can_id=0x10, name="M", signals=[a, b], dlc=8)
        data = msg.encode_from_values({"a": 50.0, "b": 150.0})
        decoded = msg.decode(data)
        assert decoded["a"] == pytest.approx(50.0)
        assert decoded["b"] == pytest.approx(150.0)


class TestFrameSchema:
    def test_csv_row_matches_documented_schema(self):
        # Pins the trace schema at the source, complementing the integration
        # test that checks the header. Row taken from the README example.
        frame = Frame(0.017290, 0x0C0, "Engine", 8, bytes.fromhex("9e0c3c0000000000"))
        assert frame.to_csv_row() == "0.017290,0x0C0,Engine,8,9e0c3c0000000000"


class TestVehicle:
    def test_by_id_maps_can_id_to_message(self):
        m1, m2 = Message(0x0C0, "Engine"), Message(0x080, "Brake")
        table = Vehicle([m1, m2]).by_id()
        assert table[0x0C0].name == "Engine"
        assert table[0x080].name == "Brake"


# --------------------------------------------------------------------------- #
# detectors.py — pure helpers                                                  #
# --------------------------------------------------------------------------- #
def _frame(t, cid=0x100):
    return Frame(t, cid, "X", 8, b"\x00" * 8)


class TestHelpers:
    def test_group_by_id_groups_and_sorts(self):
        frames = [_frame(0.3, 1), _frame(0.1, 1), _frame(0.2, 2)]
        grouped = group_by_id(frames)
        assert grouped[1] == [0.1, 0.3]      # sorted within an ID
        assert grouped[2] == [0.2]

    def test_window_counts_uses_full_windows_only(self):
        # W=0.5, T=1.0 -> two full windows [0,0.5), [0.5,1.0).
        # A timestamp at exactly T=1.0 falls in window index 2 and is dropped.
        counts = window_counts([0.0, 0.1, 0.3, 0.6, 0.9, 1.0], W=0.5, T=1.0)
        assert counts == [3, 2]

    def test_merge_combines_overlapping_same_id_same_detector(self):
        d1 = Detection(1, 0.0, 1.0, 0.0, "timing", "a")
        d2 = Detection(1, 0.5, 2.0, 0.5, "timing", "b")
        out = merge([d1, d2])
        assert len(out) == 1
        assert out[0].t_end == 2.0
        assert set(out[0].reason.split("+")) == {"a", "b"}

    def test_merge_keeps_distant_spans_separate(self):
        d1 = Detection(1, 0.0, 1.0, 0.0, "timing", "a")
        d2 = Detection(1, 2.0, 3.0, 2.0, "timing", "b")   # gap 1.0 > gap_tol 0.5
        assert len(merge([d1, d2])) == 2

    def test_merge_does_not_cross_detectors(self):
        d1 = Detection(1, 0.0, 1.0, 0.0, "timing", "a")
        d2 = Detection(1, 0.0, 1.0, 0.0, "value", "b")    # same span, different detector
        assert len(merge([d1, d2])) == 2


# --------------------------------------------------------------------------- #
# detectors.py — TimingDetector                                                #
# --------------------------------------------------------------------------- #
class TestTimingDetector:
    CID = 0x100
    DT = 0.05

    def _clean(self):
        # 40 frames at 20 Hz over ~2 s: median 5 frames per 0.25 s window.
        return [_frame(i * self.DT, self.CID) for i in range(40)]

    def test_watchdog_fires_at_last_seen_plus_gap_max(self):
        clean = self._clean()
        # Drop two consecutive frames (indices 20,21) -> one 0.15 s gap, while
        # the affected window keeps 3 frames so the count band stays quiet.
        faulted = clean[:20] + clean[22:]

        td = TimingDetector(window_s=0.25)
        td.fit(clean)
        dets = td.predict(faulted)

        assert len(dets) == 1                       # only the watchdog, nothing else
        d = dets[0]
        assert "gap" in d.reason
        expected_fire = clean[19].timestamp + td.gap_max[self.CID]
        assert d.fire_time == pytest.approx(expected_fire)
        assert d.t_start == pytest.approx(d.fire_time)   # span starts at the deadline breach

    def test_count_band_catches_silence(self):
        clean = self._clean()
        # Remove an entire window's worth of frames -> that window collapses to 0.
        faulted = [f for i, f in enumerate(clean) if not (20 <= i <= 24)]
        td = TimingDetector(window_s=0.25)
        td.fit(clean)
        dets = td.predict(faulted)
        assert any("silence" in d.reason for d in dets)

    def test_count_band_catches_burst(self):
        clean = self._clean()
        extra = [_frame(1.0 + i * 0.001, self.CID) for i in range(30)]  # all in one window
        td = TimingDetector(window_s=0.25)
        td.fit(clean)
        dets = td.predict(clean + extra)
        assert any("burst" in d.reason for d in dets)


# --------------------------------------------------------------------------- #
# detectors.py — ValueDetector                                                 #
# --------------------------------------------------------------------------- #
class TestValueDetector:
    CID = 0x200

    def _setup(self):
        sig = Signal("v", start_byte=0, length_bytes=2, scale=0.1)
        msg = Message(self.CID, "BMS", signals=[sig], dlc=8)
        vehicle = Vehicle([msg])
        clean = [
            Frame(i * 0.1, self.CID, "BMS", 8, msg.encode_from_values({"v": v}))
            for i, v in enumerate([380.0, 382.0, 384.0, 386.0, 388.0, 390.0])
        ]
        det = ValueDetector(vehicle, margin_frac=0.05)
        det.fit(clean)   # learned band ~ (379.5, 390.5): range 10, margin 0.5
        return det, msg

    def _frame(self, msg, value):
        return Frame(0.0, self.CID, "BMS", 8, msg.encode_from_values({"v": value}))

    def test_in_range_value_not_flagged(self):
        det, msg = self._setup()
        assert det.predict([self._frame(msg, 385.0)]) == []

    def test_out_of_range_value_flagged(self):
        det, msg = self._setup()
        dets = det.predict([self._frame(msg, 395.0)])
        assert len(dets) == 1
        assert "value:v" in dets[0].reason

    def test_small_drift_within_margin_is_missed(self):
        # 390.4 is above the clean max (390) but inside the learned margin band.
        # This is the documented blind spot ("catches +4 V, misses +2 V"),
        # a property of the design — the test guards it stays intentional.
        det, msg = self._setup()
        assert det.predict([self._frame(msg, 390.4)]) == []

    def test_drift_beyond_margin_is_flagged(self):
        det, msg = self._setup()
        assert len(det.predict([self._frame(msg, 391.0)])) == 1


# --------------------------------------------------------------------------- #
# diagnosis.py — classification and severity rules                             #
# --------------------------------------------------------------------------- #
class TestClassify:
    @pytest.mark.parametrize("reason,detector,expected", [
        ("count_low (silence)", "timing", "SILENCE"),
        ("gap/silence",         "timing", "SILENCE"),   # silence dominates over a gap
        ("value:pack_voltage out of range", "value", "VALUE"),
        ("anything",            "value",  "VALUE"),      # detector==value branch
        ("count_high (burst)",  "timing", "BURST"),
        ("gap/delay",           "timing", "DELAY"),
        ("mystery",             "timing", "UNKNOWN"),
    ])
    def test_classify(self, reason, detector, expected):
        d = Detection(0x100, 0.0, 1.0, 0.0, detector, reason)
        assert _classify(d) == expected


class TestSeverity:
    @pytest.mark.parametrize("fault_class,criticality,duration,expected", [
        ("SILENCE", "low",      1.0, "Low"),
        ("SILENCE", "high",     1.0, "High"),
        ("SILENCE", "critical", 1.0, "High"),
        ("BURST",   "low",      1.0, "High"),   # a babbling node degrades the whole bus
        ("BURST",   "critical", 5.0, "High"),
        ("DELAY",   "critical", 2.0, "High"),   # sustained (>=2 s)
        ("DELAY",   "critical", 1.0, "Medium"), # brief
        ("DELAY",   "low",      5.0, "Low"),
        ("DELAY",   "high",     5.0, "Medium"),
        ("VALUE",   "critical", 1.0, "High"),
        ("VALUE",   "low",      1.0, "Low"),
        ("VALUE",   "high",     1.0, "Medium"),
        ("UNKNOWN", "high",     1.0, "Medium"),
    ])
    def test_severity(self, fault_class, criticality, duration, expected):
        assert _severity(fault_class, criticality, duration) == expected


class TestDiagnose:
    def _vehicle(self):
        return Vehicle([
            Message(0x0C0, "Engine"), Message(0x080, "Brake"),
            Message(0x200, "BMS"),    Message(0x100, "Steering"),
        ])

    def test_maps_detection_to_dtc_and_orders_by_severity(self):
        silence_on_brake = Detection(0x080, 30.0, 38.0, 30.0, "timing", "count_low (silence)")
        value_on_bms = Detection(0x200, 40.8, 46.8, 46.8, "value", "value:pack_voltage out of range")

        diags = diagnose([value_on_bms, silence_on_brake], self._vehicle())

        # Most serious first: Brake silence (High) before BMS value (Medium).
        assert diags[0].code == "U-COMM-LOSS"
        assert diags[0].severity == "High"
        assert diags[1].code == "P-SIG-RANGE"
        assert diags[1].severity == "Medium"
        # VALUE headline names the offending signal, and evidence is traceable.
        assert "pack_voltage" in diags[1].headline
        assert "count_low" in diags[0].evidence


# --------------------------------------------------------------------------- #
# evaluate.py                                                                  #
# --------------------------------------------------------------------------- #
class TestOverlap:
    def test_true_when_spans_intersect(self):
        assert _overlap(1.0, 2.0, 1.5, 3.0, tol=0.0) is True

    def test_false_when_disjoint(self):
        assert _overlap(1.0, 2.0, 3.0, 4.0, tol=0.0) is False

    def test_tolerance_bridges_a_small_gap(self):
        assert _overlap(1.0, 2.0, 2.2, 3.0, tol=0.3) is True
        assert _overlap(1.0, 2.0, 2.2, 3.0, tol=0.1) is False


class TestEvaluate:
    LABEL = {
        "target_id_int": 0x080, "target_id": "0x080", "target_name": "Brake",
        "fault_type": "delay", "t_start": 30.0, "t_end": 38.0,
    }

    def test_hit_gives_perfect_scores_and_latency(self):
        hit = Detection(0x080, 30.5, 37.0, 30.7, "timing", "gap/delay")
        per_fault, summary = evaluate([hit], [self.LABEL])
        assert per_fault[0]["detected"] is True
        assert per_fault[0]["by"] == ["timing"]
        assert per_fault[0]["latency"] == pytest.approx(0.7)
        assert summary["recall"] == 1.0
        assert summary["precision"] == 1.0

    def test_spurious_detection_lowers_precision_only(self):
        hit = Detection(0x080, 30.5, 37.0, 30.7, "timing", "gap/delay")
        spurious = Detection(0x999, 5.0, 6.0, 5.0, "timing", "noise")
        _, summary = evaluate([hit, spurious], [self.LABEL])
        assert summary["recall"] == 1.0
        assert summary["precision"] == pytest.approx(0.5)
        assert summary["fp_regions"] == 1

    def test_wrong_id_is_a_miss(self):
        wrong = Detection(0x999, 30.5, 37.0, 30.7, "timing", "gap/delay")
        per_fault, summary = evaluate([wrong], [self.LABEL])
        assert per_fault[0]["detected"] is False
        assert summary["recall"] == 0.0

    def test_early_detection_clamps_latency_to_zero(self):
        early = Detection(0x080, 29.5, 31.0, 29.8, "timing", "gap/delay")
        per_fault, _ = evaluate([early], [self.LABEL])
        assert per_fault[0]["latency"] == 0.0

    def test_empty_guards(self):
        # No detections -> precision defaults to 1.0; no labels -> recall 1.0.
        _, s1 = evaluate([], [self.LABEL])
        assert s1["precision"] == 1.0 and s1["recall"] == 0.0
        hit = Detection(0x080, 30.5, 37.0, 30.7, "timing", "gap/delay")
        _, s2 = evaluate([hit], [])
        assert s2["recall"] == 1.0
