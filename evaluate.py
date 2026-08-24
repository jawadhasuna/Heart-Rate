r"""How good is the model, really? Per-patient, not just averaged.

Run:  .venv\Scripts\python evaluate.py
"""

import numpy as np
import torch
from train import BeatNet          # reuse the architecture

data = np.load("beats.npz", allow_pickle=True)
CLASSES = [str(c) for c in data["classes"]]
ckpt = torch.load("model.pt", weights_only=False)

model = BeatNet(len(CLASSES), ckpt["n_timing"])
model.load_state_dict(ckpt["state"])
model.eval()

X = torch.tensor(data["X_test"]).unsqueeze(1)
T = (torch.tensor(data["T_test"]) - ckpt["t_mean"]) / ckpt["t_std"]
y = torch.tensor(data["y_test"]).long()
src = data["src_test"]

with torch.no_grad():
    pred = torch.cat([model(X[i:i+1024], T[i:i+1024]).argmax(1)
                      for i in range(0, len(X), 1024)])

print("per patient - how many of their abnormal beats did it catch?\n")
print(f"  {'rec':>5} {'beats':>7} {'abnormal':>9} {'caught':>8} {'false alarms':>13}")
rows = []
for rec in sorted(set(src.tolist())):
    m = src == rec
    yy, pp = y[m], pred[m]
    abnormal = (yy != 0)
    n_ab = int(abnormal.sum())
    caught = int(((pp != 0) & abnormal).sum())
    false_alarm = int(((pp != 0) & ~abnormal).sum())
    recall = caught / n_ab if n_ab else float("nan")
    rows.append((rec, int(m.sum()), n_ab, recall, false_alarm))
    r = f"{recall:7.1%}" if n_ab else "      -"
    print(f"  {rec:>5} {int(m.sum()):7,} {n_ab:9,} {r} {false_alarm:13,}")

with_ab = [r for r in rows if r[2] >= 30]
best = max(with_ab, key=lambda r: r[3])
worst = min(with_ab, key=lambda r: r[3])
print(f"\n  best  patient {best[0]}: {best[3]:.1%} of {best[2]:,} abnormal beats caught")
print(f"  worst patient {worst[0]}: {worst[3]:.1%} of {worst[2]:,} abnormal beats caught")

# The clinically interesting one: ventricular beats (PVCs). This is what a
# real heart monitor is mostly watching for.
v_actual, v_said = (y == 2), (pred == 2)
print(f"\nventricular beats only")
print(f"  caught          {float((v_said & v_actual).sum()) / int(v_actual.sum()):.1%}"
      f"  ({int((v_said & v_actual).sum()):,} of {int(v_actual.sum()):,})")
print(f"  when it says V  {float((v_said & v_actual).sum()) / int(v_said.sum()):.1%} right")
