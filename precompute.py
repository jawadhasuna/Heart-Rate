r"""Run the model over a handful of records and write what the website needs.

Run:  .venv\Scripts\python precompute.py

For each record, writes into site_data/:
  <rec>.bin   the waveform, downsampled to 180 Hz, as int16
  <rec>.json  every beat: where it is, what the model said, what it really was
  index.json  the list of records, for the dropdown
"""

import json
import os
import numpy as np
import torch
import wfdb

from build_dataset import (BEFORE, AFTER, CLASSES, CLASS_NAMES,
                           SYMBOL_TO_CLASS, TEST_RECORDS)
from train import BeatNet

# Twelve patients the model has never trained on. 232 and 213 are included
# deliberately: they are the ones it does worst on, and a demo that only
# shows wins is a brochure, not a result.
SHOW = ["100", "105", "200", "202", "210", "213",
        "214", "219", "221", "228", "232", "233"]

OUT = "site_data"
DISPLAY_HZ = 180          # half the original rate; plenty for drawing a trace

os.makedirs(OUT, exist_ok=True)

ckpt = torch.load("model.pt", weights_only=False)
model = BeatNet(len(CLASSES), ckpt["n_timing"])
model.load_state_dict(ckpt["state"])
model.eval()


def analyze(record_name):
    path = f"data/mitdb/{record_name}"
    record = wfdb.rdrecord(path, channels=[0])
    ann = wfdb.rdann(path, "atr")
    signal = record.p_signal[:, 0].astype(np.float32)
    fs = record.fs

    peaks = ann.sample
    gaps = np.diff(peaks) / fs
    average_gap = np.median(gaps) if len(gaps) else 1.0

    clips, timings, truths, positions = [], [], [], []
    for i, (peak, symbol) in enumerate(zip(peaks, ann.symbol)):
        cls = SYMBOL_TO_CLASS.get(symbol)
        if cls is None or i == 0 or i >= len(peaks) - 1:
            continue
        start, end = peak - BEFORE, peak + AFTER
        if start < 0 or end > len(signal):
            continue
        clip = signal[start:end] - signal[start:end].mean()
        spread = clip.std()
        if spread < 1e-6:
            continue
        clips.append(clip / spread)
        timings.append([gaps[i-1] / average_gap, gaps[i] / average_gap,
                        gaps[i-1], gaps[i]])
        truths.append(CLASSES.index(cls))
        positions.append(int(peak))

    X = torch.tensor(np.array(clips, dtype=np.float32)).unsqueeze(1)
    T = (torch.tensor(np.array(timings, dtype=np.float32)) - ckpt["t_mean"]) / ckpt["t_std"]
    with torch.no_grad():
        logits = torch.cat([model(X[i:i+1024], T[i:i+1024])
                            for i in range(0, len(X), 1024)])
        probs = torch.softmax(logits, dim=1)
        said = probs.argmax(1).tolist()
        certainty = probs.max(1).values.tolist()

    truths = np.array(truths)
    said_arr = np.array(said)

    # The trace the browser draws. Downsampling by taking every Nth sample
    # would clip the sharp peaks; taking the most extreme value in each pair
    # keeps the spikes intact, which is what you actually want to look at.
    step = round(fs / DISPLAY_HZ)
    usable = (len(signal) // step) * step
    blocks = signal[:usable].reshape(-1, step)
    extreme = np.where(np.abs(blocks.max(1)) >= np.abs(blocks.min(1)),
                       blocks.max(1), blocks.min(1))
    # int16 holds the signal to about 0.0003 mV -- far finer than the
    # electrode noise, and a quarter the size of sending floats.
    scale = 3000.0
    wave = np.clip(extreme * scale, -32768, 32767).astype(np.int16)
    wave.tofile(f"{OUT}/{record_name}.bin")

    abnormal_true = truths != 0
    abnormal_said = said_arr != 0
    caught = int((abnormal_said & abnormal_true).sum())
    missed = int((~abnormal_said & abnormal_true).sum())
    false_alarm = int((abnormal_said & ~abnormal_true).sum())

    beats = [{
        "at": round(p / step),              # index into the downsampled trace
        "said": int(s),
        "true": int(t),
        "sure": round(float(c), 3),
    } for p, s, t, c in zip(positions, said, truths, certainty)]

    counts = {CLASSES[i]: int((said_arr == i).sum()) for i in range(len(CLASSES))}

    return {
        "record": record_name,
        "hz": fs / step,
        "samples": len(wave),
        "minutes": round(len(signal) / fs / 60, 1),
        "scale": scale,
        "beats": beats,
        "counts": counts,
        "total": len(beats),
        "abnormalSaid": int(abnormal_said.sum()),
        "abnormalTrue": int(abnormal_true.sum()),
        "caught": caught,
        "missed": missed,
        "falseAlarm": false_alarm,
        "agreement": round(float((said_arr == truths).mean()), 4),
    }


index = []
print(f"{'rec':>5} {'beats':>7} {'model says':>11} {'truth':>7} {'caught':>8} {'agree':>7}")
for name in SHOW:
    info = analyze(name)
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(info, f, separators=(",", ":"))
    recall = info["caught"] / info["abnormalTrue"] if info["abnormalTrue"] else 1.0
    index.append({
        "record": name,
        "minutes": info["minutes"],
        "total": info["total"],
        "abnormalTrue": info["abnormalTrue"],
        "caught": info["caught"],
        "recall": round(recall, 4),
        "agreement": info["agreement"],
    })
    print(f"{name:>5} {info['total']:7,} {info['abnormalSaid']:11,} "
          f"{info['abnormalTrue']:7,} {recall:7.1%} {info['agreement']:7.1%}")

with open(f"{OUT}/index.json", "w") as f:
    json.dump({"records": index, "classes": CLASSES,
               "classNames": CLASS_NAMES}, f, indent=1)

size = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e6
print(f"\nwrote {len(SHOW)} records to {OUT}/  ({size:.1f} MB total)")
