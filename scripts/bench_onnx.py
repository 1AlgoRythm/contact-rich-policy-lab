"""Export the best variant of each policy kind to ONNX, check parity with PyTorch, time one control step on CPU.

For the diffusion policy only the denoiser network is exported; the DDIM loop (clamping, noise schedule) runs in numpy
and calls ONNX Runtime once per denoising step, and the WHOLE loop is timed. -> results/onnx.json, models/onnx/*.onnx
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clpl.data import load_demos  # noqa: E402
from clpl.diffusion import ddim_sample  # noqa: E402
from clpl.policies import ACT_DIM, K_CHUNK, N_EXEC  # noqa: E402
from scripts.evaluate import load_agent  # noqa: E402

BUDGET_MS = 1000 / 20  # 20 Hz control period
STEP_COUNTS = [1, 2, 5, 10, 20, 50, 100]
N_TIMED, N_WARM = 500, 50


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def export(model, args, path, in_names, out_names):
    dyn = {n: {0: "batch"} for n in in_names + out_names}
    torch.onnx.export(model, args, str(path), input_names=in_names, output_names=out_names, dynamic_axes=dyn,
                      opset_version=17, dynamo=False)


def stats(ms):
    ms = np.asarray(ms)
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max()),
            "n": len(ms)}


def time_fn(fn, n=N_TIMED, warm=N_WARM):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return stats(out)


def main():
    torch.set_num_threads(1)
    Path("models/onnx").mkdir(parents=True, exist_ok=True)
    summ = json.loads(Path("results/summary.json").read_text())
    obs_all, _, _ = load_demos()
    rng = np.random.default_rng(0)
    res = {"budget_ms": BUDGET_MS, "machine": platform.platform(), "cpu_count": os.cpu_count(),
           "onnxruntime": ort.__version__, "torch": torch.__version__, "loadavg_before": list(os.getloadavg()),
           "exporter": "torch.onnx.export legacy TorchScript exporter (dynamo=False), opset 17", "policies": {}}
    for kind in ("mlp", "transformer", "diffusion"):
        ft = summ["variants"][f"{kind}_ft"]["0.003"]["success"] >= summ["variants"][f"{kind}_noft"]["0.003"]["success"]
        variant = f"{kind}_{'ft' if ft else 'noft'}"
        ag = load_agent(variant + "_s0")
        model = ag.model.eval()
        D = len(ag.mean)
        norm = lambda raw: ((raw[:, :D] - ag.mean) / ag.std).astype(np.float32)  # demo obs columns are [8 obs | 6 F/T]
        real = norm(obs_all[rng.choice(len(obs_all), 1000, replace=False)])
        gauss = rng.standard_normal((1000, D)).astype(np.float32)
        path = Path(f"models/onnx/{variant}_s0.onnx")
        entry = {"variant": variant, "seed": 0, "params": sum(p.numel() for p in model.parameters())}
        if kind == "diffusion":
            net = model.net
            export(net, (torch.randn(1, K_CHUNK, ACT_DIM), torch.zeros(1, dtype=torch.int64), torch.randn(1, D)), path,
                   ["x", "t", "obs"], ["eps"])
            s1 = session(path, 1)
            xs = rng.standard_normal((1000, K_CHUNK, ACT_DIM)).astype(np.float32)
            ts = rng.integers(0, model.sched.T, 1000).astype(np.int64)
            with torch.no_grad():
                ref = net(torch.from_numpy(xs), torch.from_numpy(ts), torch.from_numpy(real)).numpy()
            got = s1.run(None, {"x": xs, "t": ts, "obs": real})[0]
            entry["parity_max_abs_diff_denoiser_1000"] = float(np.abs(ref - got).max())
            # full sampling loop, same starting noise, 100 observations
            xT = torch.from_numpy(rng.standard_normal((100, K_CHUNK, ACT_DIM)).astype(np.float32))
            with torch.no_grad():
                ref_chunk = model.sample(torch.from_numpy(real[:100]), x_T=xT, steps=10).numpy()
            eps_fn = lambda x, t, o: s1.run(None, {"x": x.astype(np.float32), "t": t, "obs": o})[0]
            got_chunk = ddim_sample(eps_fn, model.sched, real[:100], (100, K_CHUNK, ACT_DIM), 10, xT.numpy())
            entry["parity_max_abs_diff_full_ddim10_loop_100"] = float(np.abs(ref_chunk - got_chunk).max())
            entry["timing"] = {}
            for threads in (1, os.cpu_count()):
                sess = session(path, threads)
                fn_eps = lambda x, t, o, sess=sess: sess.run(None, {"x": x.astype(np.float32), "t": t, "obs": o})[0]
                one = real[:1]
                per_steps = {}
                for S in STEP_COUNTS:
                    g = np.random.default_rng(1)
                    f = lambda S=S, g=g: ddim_sample(fn_eps, model.sched, one, (1, K_CHUNK, ACT_DIM), S,
                                                     g.standard_normal((1, K_CHUNK, ACT_DIM)).astype(np.float32))
                    per_steps[str(S)] = time_fn(f, n=N_TIMED if S <= 50 else 200)
                one_call = time_fn(lambda: fn_eps(np.zeros((1, K_CHUNK, ACT_DIM), np.float32), np.array([50]), one))
                entry["timing"][f"threads_{threads}"] = {"ddim_loop_by_steps": per_steps, "one_denoiser_call": one_call}
            # PyTorch eager, same loop, for comparison
            g = torch.Generator().manual_seed(1)
            o1 = torch.from_numpy(real[:1])
            entry["timing"]["pytorch_eager_1thread_ddim10"] = time_fn(lambda: model.sample(o1, generator=g, steps=10))
        else:
            export(model, (torch.randn(1, D),), path, ["obs"], ["chunk"])
            s1 = session(path, 1)
            for name, x in (("demo_obs", real), ("gaussian_obs", gauss)):
                with torch.no_grad():
                    ref = model(torch.from_numpy(x)).numpy()
                got = s1.run(None, {"obs": x})[0]
                entry[f"parity_max_abs_diff_{name}_1000"] = float(np.abs(ref - got).max())
            entry["timing"] = {}
            for threads in (1, os.cpu_count()):
                sess = session(path, threads)
                one = real[:1]
                entry["timing"][f"threads_{threads}"] = time_fn(lambda sess=sess: sess.run(None, {"obs": one}))
            o1 = torch.from_numpy(real[:1])
            with torch.no_grad():
                entry["timing"]["pytorch_eager_1thread"] = time_fn(lambda: model(o1))
        entry["onnx_size_kb"] = path.stat().st_size / 1024
        res["policies"][kind] = entry
        print(kind, variant, {k: v for k, v in entry.items() if k.startswith("parity")}, flush=True)
    res["loadavg_after"] = list(os.getloadavg())
    Path("results/onnx.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
