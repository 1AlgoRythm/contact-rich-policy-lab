"""Fill the Results block of README.md from results/*.json. Nothing in that block is typed by hand."""
import glob
import json
import re
from pathlib import Path


def load(p):
    f = Path(p)
    return json.loads(f.read_text()) if f.exists() else {}


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pct(b):
    return f"{100 * b['success']:.1f}% [{100 * b['wilson95'][0]:.1f}, {100 * b['wilson95'][1]:.1f}]"


NAMES = {"mlp": "MLP (BC)", "transformer": "Transformer chunk", "diffusion": "Diffusion chunk"}


def numbers(S, D, O, tr):
    """Values for the <!--n:key-->...<!--/n--> markers in docs/STUDY.md."""
    f = lambda b: f"{100*b['success']:.1f}% [{100*b['wilson95'][0]:.1f}, {100*b['wilson95'][1]:.1f}]"
    n = {"expert_rate": f({"success": D["expert_success_rate"], "wilson95": D["expert_success_wilson95"]})}
    for tag, key in (("3", "0.003"), ("5", "0.005"), ("8", "0.008")):
        n[f"base_{tag}"] = f(S["controllers"]["straight_to_estimate"][key])
        n[f"expert_{tag}"] = f(S["controllers"]["expert"][key])
        for v, d in S["variants"].items():
            n[f"{v}_{tag}"] = f(d[key])
    for kind, z in S["ftzero"].items():
        n["zero_" + kind.replace("_ft", "")] = f(z["0.003"])
    for k, v in S["paired_sigma0.003"].items():
        lab = k.replace(": ft minus noft", "").replace(" ", "_")
        key = ("gain_" + lab) if ": ft minus noft" in k else lab
        n[key] = f"{100*v['diff']:+.1f} points (95% interval {100*v['ci95'][0]:+.1f} to {100*v['ci95'][1]:+.1f})"
    if O:
        ms = lambda l, d=1: f"{l['median_ms']:.{d}f} / {l['p95_ms']:.{d}f}"
        for kind in ("mlp", "transformer"):
            n[f"onnx_{kind}"] = ms(O["policies"][kind]["timing"]["threads_1"], 3)
        dl = O["policies"]["diffusion"]["timing"]["threads_1"]["ddim_loop_by_steps"]
        for st in (1, 10, 50):
            n[f"onnx_diffusion{st}"] = ms(dl[str(st)], 2)
        n["parity_max"] = f"{max(v for e in O['policies'].values() for k, v in e.items() if k.startswith('parity')):.1e}"
    return n


def fill_study(n):
    p = Path("docs/STUDY.md")
    t = p.read_text()
    missing = []

    def rep(m):
        if m.group(1) not in n:
            missing.append(m.group(1))
            return m.group(0)
        return f"<!--n:{m.group(1)}-->{n[m.group(1)]}<!--/n-->"

    p.write_text(re.sub(r"<!--n:(\w+)-->.*?<!--/n-->", rep, t))
    if missing:
        print("STUDY.md keys without a value:", missing)


def main():
    S, E, D = load("results/summary.json"), load("results/env_check.json"), load("results/demos.json")
    O, DD, TM = load("results/onnx.json"), load("results/ddim_steps.json"), load("results/timings.json")
    tr = {Path(p).stem: json.loads(Path(p).read_text()) for p in sorted(glob.glob("results/train/*.json"))}
    parts = []
    # environment
    st, ft = E["stability"], E["ft_press_test"]
    parts.append("### Simulator checks (`results/env_check.json`)\n\n" + table(
        ["check", "value"],
        [["saturated random actions: control steps / non-finite observations", f"{st['control_steps']} / {st['nonfinite_obs']}"],
         ["max tip speed in those runs (m/s)", f"{st['max_abs_velocity_m_s']:.3f}"],
         ["F/T on a flat press: expected |Fz| = 2000 N/m x 5 mm (N) / measured", f"{ft['expected_fz_N']:.1f} / {abs(ft['measured_fz_N']):.2f}"],
         ["MuJoCo version", E["mujoco"]]]))
    rows = [[k.replace("_", " "), pct(v)] for k, v in E["baseline_vs_geometry_sigma3mm"].items()]
    parts.append("Straight-to-estimate baseline vs hole geometry (sigma = 3 mm, 100 episodes each). The chosen geometry is "
                 "clearance 1 mm, chamfer 4 mm.\n\n" + table(["geometry", "baseline success % [Wilson 95]"], rows))
    parts.append(f"### Demonstrations (`results/demos.json`)\n\nExpert success over all attempts: {D['demos_kept']} of {D['attempts']} "
                 f"({pct({'success': D['expert_success_rate'], 'wilson95': D['expert_success_wilson95']})}). Kept {D['samples']} "
                 f"(observation, action) pairs, mean {D['mean_len']:.1f} steps per demo.")
    # main table
    rows = []
    for name, b in (("straight to estimate (baseline)", S["controllers"]["straight_to_estimate"]["0.003"]),
                    ("scripted expert (privileged)", S["controllers"]["expert"]["0.003"])):
        rows.append([name, "-", pct(b), b["pooled_n"], f"{b['steps_to_insert_mean']:.1f}" if b["steps_to_insert_mean"] else "-",
                     f"{b['peak_force_mean_N']:.1f}", f"{b['peak_force_p95_N']:.1f}"])
    for kind in NAMES:
        for tag, lab in (("ft", "with F/T"), ("noft", "no F/T")):
            b = S["variants"][f"{kind}_{tag}"]["0.003"]
            rows.append([f"{NAMES[kind]}, {lab}", f"{100*b['seed_mean']:.1f} +/- {100*b['seed_std']:.1f}", pct(b), b["pooled_n"],
                         f"{b['steps_to_insert_mean']:.1f}" if b["steps_to_insert_mean"] else "-",
                         f"{b['peak_force_mean_N']:.1f}", f"{b['peak_force_p95_N']:.1f}"])
    parts.append("### Main result: hole-position noise 3 mm (the training noise), 200 shared eval episodes per seed\n\n" + table(
        ["controller", "mean +/- std over 3 seeds (%)", "pooled success % [Wilson 95]", "pooled episodes", "steps to insert (successes)",
         "peak force mean (N)", "peak force p95 (N)"], rows) + "\n\nOne control step = 50 ms. The pooled Wilson interval treats the 3 x 200 "
         "episodes as independent, but the same 200 eval episodes are reused by every seed, so it is somewhat optimistic; "
         "the std over seeds is the better measure of training-seed variation.")
    # robustness
    rows = []
    for name in ("straight_to_estimate", "expert"):
        c = S["controllers"][name]
        rows.append([name] + [pct(c[s]) for s in ("0.003", "0.005", "0.008")])
    for kind in NAMES:
        for tag, lab in (("ft", "with F/T"), ("noft", "no F/T")):
            v = S["variants"][f"{kind}_{tag}"]
            rows.append([f"{NAMES[kind]}, {lab}"] + [pct(v[s]) for s in ("0.003", "0.005", "0.008")])
    parts.append("### Robustness probe: hole-estimate noise beyond the training range (success % [Wilson 95])\n\n" + table(
        ["controller", "3 mm (trained)", "5 mm", "8 mm"], rows))
    # paired
    rows = [[k, f"{100*v['diff']:+.1f}", f"[{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}]"] for k, v in S["paired_sigma0.003"].items()]
    parts.append("### Paired differences at 3 mm (percentage points, bootstrap over the 200 shared episodes, success averaged over seeds)\n\n"
                 + table(["comparison (A minus B)", "difference", "95% interval"], rows))
    # negative control
    rows = []
    for kind in NAMES:
        z, n = S["ftzero"][f"{kind}_ft"]["0.003"], S["variants"][f"{kind}_ft"]["0.003"]
        rows.append([NAMES[kind], pct(n), pct(z)])
    parts.append("### Negative control: F/T-trained policies with the F/T reading blanked to zeros at test time (3 mm)\n\n" + table(
        ["policy", "normal", "F/T zeroed"], rows))
    # ddim
    if DD:
        steps = sorted({int(k.split("ddim")[1]) for k in DD if "ddim" in k})
        lat = O.get("policies", {}).get("diffusion", {}).get("timing", {}).get("threads_1", {}).get("ddim_loop_by_steps", {})
        rows = []
        for s in steps:
            row = [s, pct(DD[f"diffusion_ft_ddim{s}"]), pct(DD[f"diffusion_noft_ddim{s}"])]
            row.append(f"{lat[str(s)]['median_ms']:.2f} / {lat[str(s)]['p95_ms']:.2f}" if str(s) in lat else "-")
            rows.append(row)
        parts.append(f"### Diffusion policy: DDIM steps vs success (3 mm, PyTorch, 3 seeds x {DD['_meta']['n_episodes_per_seed']} episodes) and ONNX latency\n\n" + table(
            ["DDIM steps", "with F/T success %", "no F/T success %", "ONNX 1-thread full loop median / p95 ms (best variant, seed 0)"], rows))
    # onnx
    if O:
        rows = []
        for kind, e in O["policies"].items():
            par = max(v for k, v in e.items() if k.startswith("parity"))
            t = e["timing"]
            if kind == "diffusion":
                for th in ("threads_1", f"threads_{O['cpu_count']}"):
                    l = t[th]["ddim_loop_by_steps"]["10"]
                    rows.append([f"diffusion ({e['variant']}), 10 DDIM steps, {th.replace('_', ' ')}", f"{par:.1e}", f"{l['median_ms']:.2f}",
                                 f"{l['p95_ms']:.2f}", "yes" if l["p95_ms"] < O["budget_ms"] else "NO", f"{e['onnx_size_kb']:.0f}"])
            else:
                for th in ("threads_1", f"threads_{O['cpu_count']}"):
                    l = t[th]
                    rows.append([f"{kind} ({e['variant']}), {th.replace('_', ' ')}", f"{par:.1e}", f"{l['median_ms']:.3f}", f"{l['p95_ms']:.3f}",
                                 "yes" if l["p95_ms"] < O["budget_ms"] else "NO", f"{e['onnx_size_kb']:.0f}"])
        parts.append(f"### On-board inference check (ONNX Runtime {O['onnxruntime']}, CPU, batch 1, 500 calls, laptop not robot computer)\n\n"
                     f"Control period at 20 Hz = {O['budget_ms']:.0f} ms. Parity = max absolute difference ONNX vs PyTorch over 1,000 inputs "
                     f"(diffusion: the denoiser on 1,000 inputs and the full 10-step loop on 100). Timing was taken with load average "
                     f"{O['loadavg_before'][0]:.1f} on {O['cpu_count']} cores (other jobs were running).\n\n" + table(
                         ["policy / threads", "parity max abs diff", "median ms", "p95 ms", f"p95 < {O['budget_ms']:.0f} ms", "ONNX size KB"], rows))
    # training
    rows = []
    for kind in NAMES:
        for tag in ("ft", "noft"):
            rs = [tr[k] for k in tr if k.startswith(f"{kind}_{tag}_s")]
            if rs:
                rows.append([f"{NAMES[kind]}, {'with' if tag == 'ft' else 'no'} F/T", rs[0]["params"], rs[0]["epochs"],
                             f"{sum(r['wall_seconds'] for r in rs)/len(rs):.0f}", f"{sum(r['val_loss'] for r in rs)/len(rs):.4f}",
                             f"{sum(r['train_loss'] for r in rs)/len(rs):.4f}"])
    parts.append("### Training (CPU, 2 threads per process, 3 seeds trained in parallel on a shared machine)\n\n" + table(
        ["policy", "parameters", "epochs", "mean wall seconds per model", "mean val loss", "mean train loss"], rows) +
        "\n\nLosses are not comparable across rows (the MLP predicts 1 action, the others 8; diffusion's is a noise-prediction loss).")
    if TM:
        parts.append("### Stage wall-clock times on this machine (`results/timings.json`)\n\n" + table(
            ["stage", "seconds", "load average after"], [[k, v["wall_seconds"], v["loadavg_after"][0]] for k, v in TM.items()]))
    fill_study(numbers(S, D, O, tr))
    block = "\n\n".join(parts)
    readme = Path("README.md").read_text()
    new = re.sub(r"(<!-- results:start -->).*?(<!-- results:end -->)", lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", readme, flags=re.S)
    Path("README.md").write_text(new)
    print(block[:500])


if __name__ == "__main__":
    main()
