# Heart Rate

A heartbeat classifier trained on the MIT-BIH Arrhythmia Database, and a site that
shows what it got right and what it got wrong.

**Live: [heartrate.vercel.app](https://heartrate.vercel.app)**

Most ECG demos show you what a model claimed. This one ships the cardiologists'
labels alongside the model's verdicts, so every flagged beat is marked as caught,
missed, or a false alarm. The failures are on the page on purpose.

---

## What it does

Pick one of 12 patients, and the site draws their half-hour ECG with the model's
verdict on every heartbeat:

| | |
|---|---|
| 🔴 caught | flagged, and it really was abnormal |
| 🟠 missed | abnormal, and the model walked straight past it |
| 🔵 false alarm | flagged, but the beat was fine |

Plus the counts and a plain-language summary generated from the record's own
statistics.

## Results

Trained on 22 patients, tested on 22 **different** patients — an inter-patient
split. This matters: papers reporting ~98% on this dataset usually shuffle beats
randomly, which puts the same patient in both halves and lets the model recognise
the person rather than the condition. Splitting by patient is harder and honest.

| Class | Beats | Caught | Precision |
|---|---:|---:|---:|
| N — normal | 44,231 | 94.4% | 96.4% |
| S — early, upper chambers | 1,836 | **30.1%** | 27.1% |
| V — early, lower chambers | 3,220 | 93.0% | 88.9% |
| F — fusion | 388 | **16.0%** | 7.2% |

**Overall accuracy 91.4%. Balanced score 58.4%.**

The balanced score is the number that matters. 90% of beats are normal, so
answering "normal" every time scores 89% and learns nothing — plain accuracy
flatters any model on this dataset.

### What it learned well, and what it didn't

It's a **strong ventricular-beat detector**: 93% caught at 88.9% precision, which
is competitive with published work on this split. `V` beats have a distinctive wide
shape *and* arrive early, so both halves of the model agree.

It's an **unreliable atrial-beat detector**: 30% on `S`. An `S` beat looks
completely normal — what makes it abnormal is arriving early. It's a timing
abnormality, and timing is a much weaker signal than shape.

Per-patient, that gap is stark:

| Patient | Abnormal beats | Caught |
|---|---:|---:|
| 221 | 396 | 100% |
| 233 | 848 | 98.8% |
| 200 | 858 | 94.2% |
| 213 | 610 | 55.1% |
| **232** | **1,381** | **21.1%** |

Patient 232 is almost entirely `S` beats and is most of the model's total failure.
It's included on the site deliberately.

## Where the model actually runs

**Not in the browser.** `precompute.py` runs the model once, offline, and writes the
results to `site_data/`. The site downloads those and draws them — no inference
happens at page load, and `model.pt` is never fetched by the page.

So the site is a *recording* of the model's output. Feeding it a new ECG would
require running the model in the browser (ONNX + onnxruntime-web) or behind an API.

## The model

`BeatNet` — 48,008 parameters, written from scratch in PyTorch. Two branches:

| Branch | Reads | Layers |
|---|---|---|
| Shape | the 360-sample waveform | 3 × (Conv1d → BatchNorm → ReLU → MaxPool), 32 → 64 → 128 filters, kernels 7/5/3 |
| Timing | 4 inter-beat gap measurements | Dense 4 → 32 → 32 |
| Head | both, concatenated | 160 → Dropout(0.3) → 64 → 5 |

Adam at 1e-3, class-weighted cross-entropy, batches of 128, 15 passes. The best pass
is kept by balanced score rather than the last — training bounces, and the lowest
training loss was not the best real result.

The timing branch was added after a first version scored 24% on `S` beats and 1.5%
on `F`. Clipping each beat around its peak threw away the very thing that makes an
`S` beat abnormal. Adding the gaps lifted `V` precision from 44% to 89% and overall
accuracy from 83% to 91%.

## Running it yourself

Download the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
and unzip it into `data/mitdb/`, then:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\pip install wfdb torch matplotlib
```

```bash
.venv\Scripts\python explore.py 208
```

```bash
.venv\Scripts\python build_dataset.py
```

```bash
.venv\Scripts\python train.py
```

```bash
.venv\Scripts\python precompute.py
```

Then serve the site — it needs `http://`, not `file://`, because it fetches JSON:

```bash
python -m http.server 5180
```

## Files

| | |
|---|---|
| `explore.py` | Opens one record and plots it with the cardiologists' labels |
| `build_dataset.py` | Cuts all 48 records into 100,669 labeled 1-second clips |
| `train.py` | The model and the training loop |
| `evaluate.py` | Per-patient scoring — averages hide that one patient carries most of the misses |
| `precompute.py` | Runs the model over 12 records, writes what the site loads |
| `index.html` `styles.css` `app.js` | The site |
| `model.pt` | Trained weights, 196 KB |

`data/` and `beats.npz` are not in the repo — one is 104 MB and not ours to
redistribute, the other is rebuildable in seconds.

## Not a medical device

A learning project. Trained on 47 people recorded between 1975 and 1979, and it
misses four out of five abnormal beats on at least one of them. Nothing here is a
diagnosis.

## Credits

ECG recordings from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
(Moody GB, Mark RG, 2005), via [PhysioNet](https://physionet.org), used under the
Open Data Commons Attribution License v1.0.

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> IEEE Eng in Med and Biol 20(3):45-50 (2001).

> Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet.
> Circulation 101(23):e215-e220 (2000).
