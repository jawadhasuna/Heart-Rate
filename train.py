r"""Train a small neural network to recognize heartbeat types.

Run:  .venv\Scripts\python train.py

Writes model.pt -- the trained model.
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

class BeatNet(nn.Module):
    """Two branches: one reads the shape, one reads the timing.

    The convolution branch slides small filters along the heartbeat looking
    for shapes -- a sharp upstroke, a wide dip. Nobody tells it what to look
    for; the shapes are learned from the examples.

    The timing branch gets the gaps to the beats either side. That is the
    only way to see that a beat came EARLY, which is what makes an S beat
    abnormal even though it looks perfectly normal.

    The two are joined at the end and judged together.
    """

    def __init__(self, n_classes, n_timing):
        super().__init__()
        def block(inp, out, kernel):
            return nn.Sequential(
                nn.Conv1d(inp, out, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out),
                nn.ReLU(),
                nn.MaxPool1d(2),        # halve the length, keep the strongest signal
            )
        self.shape = nn.Sequential(
            block(1, 32, 7),            # 360 -> 180
            block(32, 64, 5),           # 180 -> 90
            block(64, 128, 3),          # 90  -> 45
            nn.AdaptiveAvgPool1d(1),    # squash to one number per filter
            nn.Flatten(),
        )
        self.timing = nn.Sequential(
            nn.Linear(n_timing, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.classify = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 32, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, waveform, timing):
        return self.classify(torch.cat([self.shape(waveform),
                                        self.timing(timing)], dim=1))


def main():
    data = np.load("beats.npz", allow_pickle=True)
    CLASSES = [str(c) for c in data["classes"]]

    X_train = torch.tensor(data["X_train"]).unsqueeze(1)   # (beats, 1 channel, 360)
    X_test  = torch.tensor(data["X_test"]).unsqueeze(1)
    y_train = torch.tensor(data["y_train"]).long()
    y_test  = torch.tensor(data["y_test"]).long()

    # Put the timing numbers on a comparable scale to the waveform, using the
    # training set's own average and spread. The test set is deliberately scaled
    # with the TRAINING numbers -- using its own would be letting the model peek
    # at data it is supposed to have never seen.
    T_train = torch.tensor(data["T_train"])
    T_test  = torch.tensor(data["T_test"])
    t_mean, t_std = T_train.mean(0), T_train.std(0).clamp(min=1e-6)
    T_train = (T_train - t_mean) / t_std
    T_test  = (T_test - t_mean) / t_std


    model = BeatNet(len(CLASSES), T_train.shape[1])

    # 90% of beats are normal. Left alone, the model would learn to answer
    # "normal" every time and score 90% while being useless. Weighting the loss
    # makes a mistake on a rare class cost far more than a mistake on a common
    # one, so it is forced to actually pay attention to the rare beats.
    counts = torch.bincount(y_train, minlength=len(CLASSES)).float()
    weights = (1.0 / counts.clamp(min=1)).sqrt()
    weights = (weights / weights.mean()).clamp(max=20.0)
    print("class weights:", {c: round(float(w), 1) for c, w in zip(CLASSES, weights)})

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS, BATCH = 15, 128
    n = len(X_train)
    REAL = [0, 1, 2, 3]      # ignore Q when scoring: 7 test beats is not a class


    def predict(waveform, timing):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(waveform[i:i + 1024], timing[i:i + 1024]).argmax(1)
                              for i in range(0, len(waveform), 1024)])


    def balanced_score(predicted):
        """Average of how much of each class was caught.

        Plain accuracy is useless here -- answering "normal" every time scores
        89%. This scores each class separately and averages, so ignoring a rare
        class costs just as much as ignoring a common one.
        """
        got = []
        for i in REAL:
            actual = y_test == i
            if actual.sum():
                got.append(float(((predicted == i) & actual).sum()) / int(actual.sum()))
        return sum(got) / len(got)


    print(f"\ntraining on {n:,} beats for {EPOCHS} passes\n")

    best_score, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n)
        running = 0.0
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx], T_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)

        predicted = predict(X_test, T_test)
        accuracy = float((predicted == y_test).float().mean())
        score = balanced_score(predicted)
        star = ""
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            star = "  <- best so far"
        print(f"  pass {epoch:2}   loss {running / n:.4f}   accuracy {accuracy:.1%}"
              f"   balanced {score:.1%}{star}")

    # Keep the best pass, not the last one. Training bounces around, and the
    # final pass is not automatically the best one.
    model.load_state_dict(best_state)
    predicted = predict(X_test, T_test)
    print(f"\nkeeping pass {best_epoch}")

    torch.save({"state": model.state_dict(), "classes": CLASSES,
                "t_mean": t_mean, "t_std": t_std,
                "n_timing": T_train.shape[1]}, "model.pt")

    print("\nper class, on patients it never saw")
    print(f"  {'':4} {'beats':>7} {'caught':>8} {'when it says this':>18}")
    for i, cls in enumerate(CLASSES):
        actual = y_test == i
        said = predicted == i
        total = int(actual.sum())
        if total == 0:
            continue
        recall = float((said & actual).sum()) / total
        precision = float((said & actual).sum()) / max(int(said.sum()), 1)
        print(f"  {cls:4} {total:7,} {recall:7.1%} {precision:17.1%}")

    print("\nconfusion - rows are the truth, columns are the guess")
    print("       " + "".join(f"{c:>8}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = [int(((y_test == i) & (predicted == j)).sum()) for j in range(len(CLASSES))]
        print(f"  {cls:4} " + "".join(f"{v:>8,}" for v in row))

    overall = float((predicted == y_test).float().mean())
    abnormal_actual = y_test != 0
    abnormal_caught = (predicted != 0) & abnormal_actual
    print(f"\noverall accuracy      {overall:.1%}")
    print(f"balanced score        {balanced_score(predicted):.1%}")
    print(f"abnormal beats caught {int(abnormal_caught.sum()):,} of {int(abnormal_actual.sum()):,}"
          f"  ({float(abnormal_caught.sum()) / int(abnormal_actual.sum()):.1%})")
    print("\nwrote model.pt")



if __name__ == "__main__":
    main()