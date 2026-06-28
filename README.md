# CAN data generator + fault injection harness (Phase 1)

A dependency-free (pure stdlib Python) harness that generates realistic CAN bus
traffic, injects a scheduled campaign of faults, and writes the ground truth
alongside it. Everything downstream — anomaly detection, monitoring, dashboard —
is built and *measured* against the labels this produces.

## Why it's built this way

Three design decisions carry the whole thing, and each maps to a question a
serious automotive engineer will ask:

1. **Virtual clock, not `time.sleep`.** Generation is a discrete-event
   simulation: each message has a "next send time", we advance to the earliest,
   emit a frame, reschedule. An hour of traffic is produced in a fraction of a
   second, deterministically. Wall-clock real-time is a *replay* concern for the
   live demo, kept separate from generation. *("How do you generate enough data,
   and is it reproducible?")*

2. **Every fault is logged ground truth.** Faults are a scheduled campaign of
   fully-specified events `(target_id, type, t_start, t_end, params)`. That list
   is written verbatim to `labels.json`. Nothing is corrupted that isn't
   recorded, so a detection at time *t* on ID *x* is a true positive iff it
   falls inside a matching window. *("How do you know your detector works?")*

3. **Generation and injection are separate stages.** The generator emits a clean
   trace; the fault layer transforms a copy of it. You get a clean file to learn
   "normal" from and a faulted file to test on, from one run. *("Where does your
   training data come from if the data is faulted?")*

## Files

| File           | Responsibility                                                  |
|----------------|-----------------------------------------------------------------|
| `model.py`     | Signals, messages, frames, byte-level encode/decode (mini-DBC). |
| `config.py`    | The vehicle: 4 ECUs + 1 event-triggered message; demo campaign. |
| `generator.py` | Virtual-clock engine producing the clean frame stream.          |
| `faults.py`    | Fault event types and campaign application (= the ground truth).|
| `run.py`       | CLI: writes clean trace, faulted trace, labels.                 |

## Run

```bash
python -m can_bus_sim.run --duration 60 --seed 0 --out-dir out
```

Outputs in `out/`:

- `clean_trace.csv` — fault-free traffic. Train your "normal" model on this.
- `faulted_trace.csv` — same traffic with the campaign applied. Test on this.
- `labels.json` — the injected faults; ground truth for scoring.

Trace schema (raw frames, the way a real CAN logger captures them — decoding is
the detector's job, via the model in `model.py`):

```
timestamp,can_id,name,dlc,data_hex
0.017290,0x0C0,Engine,8,9e0c3c0000000000
```

## Fault types (mapped to the anomaly taxonomy)

| Taxonomy        | Fault type(s)                        | What it does                              |
|-----------------|--------------------------------------|-------------------------------------------|
| Missing         | `silence`, `intermittent_drop`       | Node goes quiet / flaky frame loss        |
| Timing          | `delay`                              | Adds latency + jitter to in-window frames |
| Abnormal value  | `value` (`stuck`/`offset`/`noise`)   | Decode → perturb one signal → re-encode   |
| Traffic spike   | `burst`                              | Extra frames at a much higher rate        |

The default 60 s campaign and a verification pass (clean vs faulted) give:

| Fault    | ID     | Effect measured in the window                          |
|----------|--------|--------------------------------------------------------|
| silence  | 0x0C0  | Engine frames 600 → **0**                              |
| burst    | 0x100  | Steering frames 101 → **2102**                         |
| value    | 0x200  | `pack_voltage` mean 385 → **445 V** (+60)              |
| delay    | 0x080  | inter-arrival stdev 0.2 → **8.1 ms**, max gap → **49 ms** |

### One subtlety worth knowing

The delay fault leaves the inter-arrival *mean* at ~10 ms. That's correct, not a
bug: a roughly constant latency applied to every frame cancels in the
differences. Constant latency is **not observable** from inter-arrival timing on
a passive bus monitor — you'd need an external time reference. Jitter (stdev) and
missed deadlines (max gap) are what's observable. So a timing detector should key
on per-ID inter-arrival **spread and deadline gaps**, not the mean.

## Simplifications (be ready to name these)

- Signals are byte-aligned little-endian; real DBC signals use arbitrary bit
  offsets. Contained entirely in `Signal.encode/decode`.
- Signal dynamics are stylised functions of time, not a vehicle physics model.
- No bus arbitration / error-frame modelling; this is an application-layer
  trace, not a bit-level CAN controller simulation.

## Phase 2 — detection + evaluation

Two interpretable detectors, each learning "normal" from `clean_trace.csv`:

| Detector        | Looks at        | Catches                |
|-----------------|-----------------|------------------------|
| `TimingDetector`| when frames arrive | silence, delay, burst |
| `ValueDetector` | what frames contain | out-of-range signals  |

The timing detector is a per-ID **watchdog** (a deadline breach fires at
`last_seen + gap_max`, so silence is caught ~one cycle late, not retroactively)
plus a per-ID per-window **count band** (collapse = silence, explosion = burst).
The value detector learns a plausibility range per `(ID, signal)`.

Run it:

```bash
python -m can_bus_sim.detect --clean out/clean_trace.csv \
    --faulted out/faulted_trace.csv --labels out/labels.json --out-dir out
```

Writes `report.txt` and `detections.json`. On the default 60 s campaign:

```
ID     name      type     window       detected  by      latency
0x0C0  Engine    silence  18.0-24.0s   YES       timing  0.014s
0x080  Brake     delay    30.0-38.0s   YES       timing  0.016s
0x200  BMS       value    40.8-46.8s   YES       value   0.054s
0x100  Steering  burst    51.0-53.0s   YES       timing  0.250s

recall 100%, precision 100%, 0 false positives on the clean trace
```

Each fault is caught by the *right* detector — the argument for layering. The
event-triggered TurnSignal (0x350) produces **zero** false positives: the
detector learns its irregular timing from clean data rather than assuming a
fixed cycle.

### Don't be fooled by 100%

The injected faults sit well above threshold, so high scores are expected. The
value detector's measured floor on `pack_voltage`: it catches a **+4 V** drift
but misses **+2 V**, because it can't resolve a drift smaller than the signal's
own clean range margin. The real result is the curve of precision/recall as you
sweep fault magnitude and add bus noise — that sweep is the next artifact, and
it's where a rate-of-change detector would earn its place alongside the range
check.

## Module map (Phase 1 + 2)

```
model.py      Signal / Message / Frame / Vehicle, encode-decode
config.py     the 4-ECU vehicle + default fault campaign
generator.py  virtual-clock clean traffic
faults.py     fault events + campaign application (= ground truth)
traceio.py    trace/label read + write (one owner of the schema)
detectors.py  TimingDetector, ValueDetector, Detection, merge
evaluate.py   overlap matching -> precision / recall / latency
run.py        CLI: generate clean + faulted + labels
detect.py     CLI: fit on clean, predict on faulted, score
```

## Next step (Phase 3)

Two directions, both short:
- **Diagnosis layer** — turn a `Detection` into the plain-language output you
  designed ("Engine ECU communication unstable — likely partial failure —
  severity medium"), by mapping detector + reason + duration to a message.
- **Sweep harness** — vary fault magnitude × type × bus-noise and plot
  recall/precision/latency, so the system has a characterised operating range
  instead of a single pass/fail.