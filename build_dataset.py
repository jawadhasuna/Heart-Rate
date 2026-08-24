r"""Cut every recording into individual labeled heartbeats.

Run:  .venv\Scripts\python build_dataset.py

Writes beats.npz — the training set the model learns from.
"""

import numpy as np
import wfdb

# The 48 records split into two groups of patients. The model trains on one
# group and is tested on the OTHER — people it has never seen. This is the
# honest way to test. If you shuffled all beats together, the same patient's
# heartbeats would land in both halves and the score would be inflated,
# because hearts are individual: learning one person's beats tells you a lot
# about their other beats, and nothing about a stranger.
TRAIN_RECORDS = ["101","106","108","109","112","114","115","116","118","119",
                 "122","124","201","203","205","207","208","209","215","220",
                 "223","230"]
TEST_RECORDS  = ["100","103","105","111","113","117","121","123","200","202",
                 "210","212","213","214","219","221","222","228","231","232",
                 "233","234"]
# Records 102, 104, 107 and 217 are left out entirely: those patients have
# pacemakers, so their beats are made by a machine, not the heart.

# The cardiologists used ~20 different symbols. The medical standard (AAMI)
# groups them into 5 meaningful classes. Anything not listed here isn't a
# heartbeat at all (rhythm markers, noise flags) and gets skipped.
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = {
    "N": "normal",
    "S": "early, from the upper chambers",
    "V": "early, from the lower chambers",
    "F": "a fusion of normal and ventricular",
    "Q": "unclassifiable or paced",
}
SYMBOL_TO_CLASS = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

BEFORE, AFTER = 130, 230        # samples kept either side of the beat's peak
WINDOW = BEFORE + AFTER         # 360 samples = exactly 1 second at 360 Hz


def beats_from(record_name):
    """Return (clips, timings, labels) for one record."""
    path = f"data/mitdb/{record_name}"
    record = wfdb.rdrecord(path, channels=[0])   # channel 0 is the MLII lead
    ann = wfdb.rdann(path, "atr")
    signal = record.p_signal[:, 0]

    # Gaps between consecutive beats, in seconds. An "early" beat is one
    # that arrives sooner after the last one than the patient's own rhythm
    # would predict -- so what matters is the gap compared to their average,
    # not the gap itself. A slow heart and a fast heart are both normal.
    peaks = ann.sample
    gaps = np.diff(peaks) / record.fs
    average_gap = np.median(gaps) if len(gaps) else 1.0

    clips, timings, labels = [], [], []
    for i, (peak, symbol) in enumerate(zip(peaks, ann.symbol)):
        cls = SYMBOL_TO_CLASS.get(symbol)
        if cls is None:
            continue                              # not a heartbeat
        if i == 0 or i >= len(peaks) - 1:
            continue                              # no gap on one side
        before_gap = gaps[i - 1]
        after_gap = gaps[i]
        start, end = peak - BEFORE, peak + AFTER
        if start < 0 or end > len(signal):
            continue                              # too close to either edge
        clip = signal[start:end]

        # Every patient's signal sits at its own baseline and amplitude —
        # different body, different electrode placement. Recentering each
        # clip on zero and scaling it lets the model compare beats across
        # people instead of learning who the patient is.
        clip = clip - clip.mean()
        spread = clip.std()
        if spread < 1e-6:
            continue                              # flat line, dead lead
        clips.append(clip / spread)
        timings.append([
            before_gap / average_gap,     # early? (below 1 means yes)
            after_gap / average_gap,      # was the next beat delayed?
            before_gap,                   # the raw gaps too, in seconds
            after_gap,
        ])
        labels.append(CLASSES.index(cls))

    return (np.array(clips, dtype=np.float32),
            np.array(timings, dtype=np.float32),
            np.array(labels, dtype=np.int8))


def collect(record_names, group):
    all_clips, all_timings, all_labels, all_sources = [], [], [], []
    for name in record_names:
        clips, timings, labels = beats_from(name)
        all_clips.append(clips)
        all_timings.append(timings)
        all_labels.append(labels)
        all_sources.append(np.full(len(labels), int(name), dtype=np.int16))
        print(f"  {name}  {len(labels):5,} beats")
    X = np.concatenate(all_clips)
    T = np.concatenate(all_timings)
    y = np.concatenate(all_labels)
    src = np.concatenate(all_sources)
    print(f"  {group}: {len(y):,} beats from {len(record_names)} patients")
    return X, T, y, src


print(f"cutting {WINDOW}-sample windows ({BEFORE} before the peak, {AFTER} after)\n")

print("training patients")
X_train, T_train, y_train, src_train = collect(TRAIN_RECORDS, "train")

print("testing patients (never seen during training)")
X_test, T_test, y_test, src_test = collect(TEST_RECORDS, "test")

np.savez_compressed(
    "beats.npz",
    X_train=X_train, T_train=T_train, y_train=y_train, src_train=src_train,
    X_test=X_test, T_test=T_test, y_test=y_test, src_test=src_test,
    classes=np.array(CLASSES),
)

print("what's in each class")
print(f"  {'':4} {'train':>9} {'test':>9}   ")
for i, cls in enumerate(CLASSES):
    n_tr, n_te = int((y_train == i).sum()), int((y_test == i).sum())
    pct = 100 * n_tr / len(y_train)
    print(f"  {cls:4} {n_tr:9,} {n_te:9,}   {pct:5.2f}%  {CLASS_NAMES[cls]}")

print(f"\nwrote beats.npz  ({X_train.shape[0]:,} train / {X_test.shape[0]:,} test)")
