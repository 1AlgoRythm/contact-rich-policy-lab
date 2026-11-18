"""Train 3 policy kinds x {with, without F/T} x 3 seeds. Saves models/<name>.pt and results/train/<name>.json
(one JSON per model so that seeds can be trained in parallel processes without write races)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clpl.data import load_demos, make_chunks, split  # noqa: E402
from clpl.policies import K_CHUNK, make_policy, n_params  # noqa: E402

EPOCHS = {"mlp": 100, "transformer": 100, "diffusion": 200}
BATCH, LR = 256, 1e-3


def train_one(kind, use_ft, seed, obs, chunks, tr, va, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols = slice(0, 14 if use_ft else 8)
    X = obs[:, cols]
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = torch.from_numpy(((X - mean) / std).astype(np.float32))
    Y = torch.from_numpy(chunks)
    model = make_policy(kind, X.shape[1])
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.995))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    tr_idx = torch.from_numpy(np.where(tr)[0])
    steps_per_epoch = int(np.ceil(len(tr_idx) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        for i in range(steps_per_epoch):
            b = perm[i * BATCH:(i + 1) * BATCH]
            loss = model.loss(Xn[b], Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ema.update_parameters(model)
    final = ema.module.eval()
    torch.manual_seed(1234)  # same noise draws for every model's validation loss
    with torch.no_grad():
        vi = torch.from_numpy(np.where(va)[0])
        vloss = float(np.mean([final.loss(Xn[vi], Y[vi]).item() for _ in range(8 if kind == "diffusion" else 1)]))
        tloss = float(final.loss(Xn[tr_idx[:2000]], Y[tr_idx[:2000]]).item())
    return final, mean, std, {"val_loss": vloss, "train_loss": tloss, "params": n_params(final)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=["mlp", "transformer", "diffusion"])
    ap.add_argument("--demos", default="data/demos.npz")
    ap.add_argument("--scale_epochs", type=float, default=1.0)
    a = ap.parse_args()
    obs, act, ep = load_demos(a.demos)
    chunks = make_chunks(obs, act, ep, K_CHUNK)
    tr, va = split(ep)
    Path("models").mkdir(exist_ok=True)
    torch.set_num_threads(2)
    Path("results/train").mkdir(parents=True, exist_ok=True)
    for kind in a.kinds:
        for use_ft in (True, False):
            for seed in a.seeds:
                name = f"{kind}_{'ft' if use_ft else 'noft'}_s{seed}"
                t0 = time.time()
                epochs = int(EPOCHS[kind] * a.scale_epochs)
                # the MLP predicts one action; the chunk policies predict K_CHUNK
                model, mean, std, info = train_one(kind, use_ft, seed, obs, chunks, tr, va, epochs)
                info.update(epochs=epochs, wall_seconds=time.time() - t0)
                info["loadavg_after"] = list(__import__("os").getloadavg())
                torch.save({"state": model.state_dict(), "kind": kind, "use_ft": use_ft, "mean": mean, "std": std},
                           f"models/{name}.pt")
                Path(f"results/train/{name}.json").write_text(json.dumps(info, indent=2))
                print(name, info, flush=True)


if __name__ == "__main__":
    main()
