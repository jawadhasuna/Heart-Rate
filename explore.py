r"""Open one MIT-BIH record and look at it.

Run:  .venv\Scripts\python explore.py 100
"""

import sys
import wfdb
import matplotlib
matplotlib.use("Agg")           # save to a file instead of opening a window
import matplotlib.pyplot as plt

record_name = sys.argv[1] if len(sys.argv) > 1 else "100"
path = f"data/mitdb/{record_name}"

# The signal. `record.p_signal` is a table: one row per sample, one column
# per channel. Reading the whole 30 minutes takes about a second.
record = wfdb.rdrecord(path)

# The cardiologists' labels. `sample` is WHERE each beat is (which sample
# number), `symbol` is WHAT it was ('N' = normal, 'V' = ventricular, ...).
ann = wfdb.rdann(path, "atr")

print(f"record {record_name}")
print(f"  channels     {record.sig_name}")
print(f"  sample rate  {record.fs} Hz")
print(f"  samples      {record.sig_len:,}  ({record.sig_len / record.fs / 60:.1f} min)")
print(f"  beats        {len(ann.sample):,}")
print()

# Count how many of each beat type the doctors marked.
counts = {}
for symbol in ann.symbol:
    counts[symbol] = counts.get(symbol, 0) + 1

print("  beat labels")
for symbol, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"    {symbol!r:5} {n:6,}")
print()

# Draw the first 10 seconds of channel 0, with every beat marked.
seconds = 10
end = seconds * record.fs
signal = record.p_signal[:end, 0]
times = [i / record.fs for i in range(end)]

beats_here = [(s, sym) for s, sym in zip(ann.sample, ann.symbol) if s < end]

plt.figure(figsize=(14, 4))
plt.plot(times, signal, linewidth=0.8, color="#444")
for s, sym in beats_here:
    normal = sym == "N"
    plt.axvline(s / record.fs, color="#bbb" if normal else "#E03131",
                linewidth=1, alpha=0.9)
    plt.text(s / record.fs, signal.max(), sym, fontsize=8, ha="center",
             color="#888" if normal else "#E03131")

plt.title(f"record {record_name} — {record.sig_name[0]}, first {seconds} seconds")
plt.xlabel("seconds")
plt.ylabel("mV")
plt.tight_layout()
out = f"record_{record_name}.png"
plt.savefig(out, dpi=110)
print(f"  wrote {out}")
