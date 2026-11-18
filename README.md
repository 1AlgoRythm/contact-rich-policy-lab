# contact-rich-policy-lab

A small MuJoCo experiment on one question:

> In a contact-rich insertion where the hole position is only known with error, do force/torque readings help a learned
> policy, and does a diffusion policy beat a plain behaviour-cloning MLP and a transformer action-chunk policy?

A square peg on a position-controlled "hand" must go into a chamfered hole. The hand carries a wrist force/torque sensor.
Three small imitation-learning policies (MLP, transformer action chunks, diffusion action chunks) are trained on scripted
demonstrations, each with and without the force/torque channels, three training seeds each, then compared on fixed shared
evaluation episodes. The best variants are exported to ONNX and timed against a 20 Hz control budget.
Everything in the **Results** section is generated from `results/*.json` by `scripts/render_readme.py`; no number there is typed by hand.

**Short answer (numbers are in Results below).** In this simulation the force/torque channel helped every policy a lot, and a control that blanks it at test time confirms the policies use it. A diffusion policy did **not** clearly beat the MLP: with force/torque and the training noise all three sit near the ceiling; without force/torque the plain MLP was best; diffusion was ahead only when I pushed the hole-estimate noise well beyond the training range. Possible reasons (not tested): a nearly single-mode scripted expert, and little data. Read the Limits before trusting any of it.

## Limits (read this first)

- **Simulation only.** No physical robot. **No real tactile sensor**: the "force/torque sensor" is MuJoCo's `force`/`torque` sensor on a welded body, plus small Gaussian noise (0.05 N, 0.002 Nm). Real sensors have drift, bias, delay, and contact that is messier than MuJoCo's soft-contact model. This is **not** a sim-to-real result and says nothing about real hardware.
- **Scripted demonstrations, not human ones.** The expert is a short state machine that is handed the TRUE hole position after the peg first touches the block (privileged information). Human demos are multimodal and noisier; this expert is not.
- **Task is deliberately small.** 3 translational degrees of freedom (no rotation), square peg, gravity off, one hole size, hole centre within +/-8 mm of the origin. These are small policies (about 135k to 220k parameters), 500 demos (about 10,000 samples), no images.
- **Small statistics.** 3 training seeds per variant and 200 evaluation episodes per seed. The same 200 evaluation episodes are used for every seed and variant (paired comparison), so pooled intervals are somewhat optimistic; the spread over seeds is reported next to them.
- **The F/T channel is not a clean "extra sense" in this sim.** The arm's setpoint persists between steps, so the F/T reading also reveals how hard the controller is pushing (a hidden state the observation otherwise lacks). I did not run an ablation that gives the policies the setpoint directly, so I cannot separate "contact information" from "setpoint information" in the F/T gain.
- **Geometry and controller were chosen by me**, once, before any policy was trained: clearance 1 mm, chamfer 4 mm, so that the straight-to-estimate baseline is neither near 0 nor near 100% (see the geometry table in Results). They were not tuned afterwards for the learned policies.
- **Timing is a laptop CPU** (Apple M4, shared with other running jobs; load average is recorded next to every timing). It is not a robot computer, and the numbers are not predictions for one.
- **Design choices cited, and what I did not check.** I read the arXiv abstract pages (not the full papers) of ACT (2304.13705), Diffusion Policy (2303.04137), DDPM (2006.11239), DDIM (2010.02502) and Improved DDPM (2102.09672), and the MuJoCo XML reference for the force/torque sensor. The cosine noise schedule formula, the use of `x0` clipping in the sampler, and the use of EMA weights are **from memory, not checked** against the papers' bodies.

## How the simulated world works (plain words)

- **Peg and hand.** A 20 mm x 20 mm x 50 mm peg is welded to a "hand" that slides along x, y, z. Each axis has a spring-and-damper position controller (stiffness 2000 N/m, damping 80 N s/m). Gravity is off.
- **Action.** At 20 Hz (one step = 50 ms = 25 physics steps of 2 ms) the policy outputs three numbers in [-1, 1]. Each moves the hand's *setpoint* by up to 4 mm. The setpoint may lead the measured position by at most 5 mm, so the hand pushes with at most 2000 x 0.005 = 10 N per axis. Because the setpoint persists, a pressed peg feels a spring force, like a stiff position-controlled arm.
- **Block.** A block with a square hole 1 mm wider than the peg on each side (22 mm hole, 20 mm peg) and a 45 degree chamfer 4 mm wide around the hole mouth, hole depth 30 mm.
- **Hole position.** The true hole centre is random within +/-8 mm of the origin. The policy sees only a noisy estimate: true position plus Gaussian noise of 3 mm per axis (fixed for the whole episode, like a calibration or vision offset). Noise is a parameter; the robustness probe raises it.
- **Observation (8 numbers, +6 with F/T).** Tip position (3), tip velocity (3), noisy hole estimate minus tip xy (2); optionally force (3) and torque (3) from the wrist sensor.
- **Success rule.** The peg tip is more than 15 mm below the block top within 100 steps (5 s). The episode ends at success or at the step limit.
- **What a miss looks like.** When the peg lands slightly off, its corner sits on the chamfer. A controller that keeps holding its (wrong) target wedges there. Letting the peg comply, i.e. moving the way the contact pushes, lets it slide in. The F/T reading shows which way the contact pushes.

## In plain words: diffusion policy, action chunking, and why force/torque might help

A **diffusion policy** generates actions the way image diffusion models generate pictures. In training, a true action sequence is mixed with random noise at a random strength, and a network learns to predict that noise given the current observation. At run time it starts from pure noise and removes the predicted noise step by step (here 10 steps) until an action sequence is left. Its appeal is that it can represent several equally good behaviours instead of averaging them (Chi et al. 2023; abstract read). **Action chunking** means the policy predicts the next K actions at once (here 8) and the robot executes the first few (here 4) before asking again; it makes motion smoother and cuts down on compounding small errors (Zhao et al. 2023; abstract read). **Force/torque** might help in insertion because vision-like position estimates are off by a few millimetres, which is more than the clearance, while the contact itself tells you directly which way the hole is: when the peg rests on the chamfer, the sideways force points toward the hole centre.

## Run

```bash
make setup                         # uv venv --python 3.12 .venv; installs requirements.txt (pinned)
make all                           # test envcheck demos train eval summarize ddim onnx readme
# or stage by stage:
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/check_env.py        # results/env_check.json   simulator checks, baseline vs geometry
.venv/bin/python scripts/gen_demos.py        # data/demos.npz, results/demos.json
sh scripts/train_all.sh                      # models/*.pt, results/train/*.json   (18 models)
.venv/bin/python scripts/evaluate.py --n 200 # results/eval.json   per-episode outcomes
.venv/bin/python scripts/summarize.py        # results/summary.json  intervals and paired comparisons
.venv/bin/python scripts/ddim_steps.py       # results/ddim_steps.json
.venv/bin/python scripts/bench_onnx.py       # results/onnx.json, models/onnx/*.onnx
.venv/bin/python scripts/render_readme.py    # rewrites the Results block below and the numbers in docs/STUDY.md
```

Stage wall times on this machine are in the Results block (`results/timings.json`, written by `scripts/timed.py`).

## Files

| path | what |
|---|---|
| `clpl/env.py` | MuJoCo model (built as XML in code) and the `InsertionEnv` class |
| `clpl/expert.py` | straight-to-estimate baseline, privileged expert, `rollout` |
| `clpl/diffusion.py` | noise schedule, forward noising, DDIM sampler (numpy) |
| `clpl/policies.py` | the three policies and the `Agent` wrapper that executes chunks |
| `clpl/data.py`, `clpl/stats.py` | demo windows; Wilson interval, paired bootstrap |
| `scripts/` | one script per stage (see Run) |
| `tests/` | pytest checks (hand-computed values, invariants, deliberate-break checks) |
| `docs/STUDY.md` | design decisions and interview questions |

## How things are measured

- **Evaluation episodes.** Episode seeds 100000 to 100199 give every controller the same hole position and the same unit-normal estimate error; the noise level only scales that error, so the 3, 5 and 8 mm runs share holes. Demo seeds are 0 to 499 (no overlap).
- **Success rate.** Wilson 95% interval on pooled episodes (3 seeds x 200). Also mean +/- std over the 3 seeds. Paired differences use a bootstrap over the 200 shared episodes (success averaged over seeds first).
- **Steps to insert** is averaged over successful episodes only. **Peak force** is the largest norm of the noise-free force over the episode, averaged over all episodes.
- **Chunk execution.** MLP: replans every step. Transformer and diffusion: predict 8 actions, execute 4, replan.
- **Training.** AdamW, learning rate 1e-3 with cosine decay, batch 256, L2 loss (epsilon-prediction for diffusion), exponential-moving-average weights (decay 0.995); 100 epochs (MLP, transformer) or 200 (diffusion). Last 50 demo episodes held out only to report validation loss; no checkpoint selection on it. Observations are standardised with training-set statistics.
- **F/T "zeroed" control.** The F/T-trained policies are evaluated with the six F/T numbers replaced by zeros. If they relied on force, success must fall.

## Results

<!-- results:start -->
### Simulator checks (`results/env_check.json`)

| check | value |
|---|---|
| saturated random actions: control steps / non-finite observations | 4962 / 0 |
| max tip speed in those runs (m/s) | 0.094 |
| F/T on a flat press: expected |Fz| = 2000 N/m x 5 mm (N) / measured | 10.0 / 10.00 |
| MuJoCo version | 4.0.1 |

Straight-to-estimate baseline vs hole geometry (sigma = 3 mm, 100 episodes each). The chosen geometry is clearance 1 mm, chamfer 4 mm.

| geometry | baseline success % [Wilson 95] |
|---|---|
| clearance 0.5mm chamfer 2mm | 19.0% [12.5, 27.8] |
| clearance 0.5mm chamfer 4mm | 19.0% [12.5, 27.8] |
| clearance 0.5mm chamfer 6mm | 19.0% [12.5, 27.8] |
| clearance 1mm chamfer 2mm | 27.0% [19.3, 36.4] |
| clearance 1mm chamfer 4mm | 25.0% [17.5, 34.3] |
| clearance 1mm chamfer 6mm | 27.0% [19.3, 36.4] |
| clearance 2mm chamfer 2mm | 57.0% [47.2, 66.3] |
| clearance 2mm chamfer 4mm | 52.0% [42.3, 61.5] |
| clearance 2mm chamfer 6mm | 57.0% [47.2, 66.3] |

### Demonstrations (`results/demos.json`)

Expert success over all attempts: 500 of 500 (100.0% [99.2, 100.0]). Kept 10141 (observation, action) pairs, mean 20.3 steps per demo.

### Main result: hole-position noise 3 mm (the training noise), 200 shared eval episodes per seed

| controller | mean +/- std over 3 seeds (%) | pooled success % [Wilson 95] | pooled episodes | steps to insert (successes) | peak force mean (N) | peak force p95 (N) |
|---|---|---|---|---|---|---|
| straight to estimate (baseline) | - | 26.5% [20.9, 33.0] | 200 | 16.2 | 11.4 | 16.1 |
| scripted expert (privileged) | - | 100.0% [98.1, 100.0] | 200 | 20.3 | 4.1 | 8.6 |
| MLP (BC), with F/T | 99.7 +/- 0.3 | 99.7% [98.8, 99.9] | 600 | 18.9 | 5.7 | 10.9 |
| MLP (BC), no F/T | 90.5 +/- 0.9 | 90.5% [87.9, 92.6] | 600 | 19.4 | 6.8 | 16.9 |
| Transformer chunk, with F/T | 99.3 +/- 0.8 | 99.3% [98.3, 99.7] | 600 | 22.0 | 6.3 | 11.0 |
| Transformer chunk, no F/T | 79.8 +/- 2.1 | 79.8% [76.4, 82.8] | 600 | 22.0 | 7.7 | 16.2 |
| Diffusion chunk, with F/T | 100.0 +/- 0.0 | 100.0% [99.4, 100.0] | 600 | 23.4 | 7.9 | 11.9 |
| Diffusion chunk, no F/T | 83.3 +/- 1.6 | 83.3% [80.1, 86.1] | 600 | 26.6 | 9.6 | 16.9 |

One control step = 50 ms. The pooled Wilson interval treats the 3 x 200 episodes as independent, but the same 200 eval episodes are reused by every seed, so it is somewhat optimistic; the std over seeds is the better measure of training-seed variation.

### Robustness probe: hole-estimate noise beyond the training range (success % [Wilson 95])

| controller | 3 mm (trained) | 5 mm | 8 mm |
|---|---|---|---|
| straight_to_estimate | 26.5% [20.9, 33.0] | 11.0% [7.4, 16.1] | 6.5% [3.8, 10.8] |
| expert | 100.0% [98.1, 100.0] | 100.0% [98.1, 100.0] | 100.0% [98.1, 100.0] |
| MLP (BC), with F/T | 99.7% [98.8, 99.9] | 95.7% [93.7, 97.0] | 85.5% [82.5, 88.1] |
| MLP (BC), no F/T | 90.5% [87.9, 92.6] | 67.0% [63.1, 70.6] | 39.0% [35.2, 43.0] |
| Transformer chunk, with F/T | 99.3% [98.3, 99.7] | 90.2% [87.5, 92.3] | 64.7% [60.8, 68.4] |
| Transformer chunk, no F/T | 79.8% [76.4, 82.8] | 56.5% [52.5, 60.4] | 33.3% [29.7, 37.2] |
| Diffusion chunk, with F/T | 100.0% [99.4, 100.0] | 99.5% [98.5, 99.8] | 88.7% [85.9, 91.0] |
| Diffusion chunk, no F/T | 83.3% [80.1, 86.1] | 58.7% [54.7, 62.5] | 30.5% [26.9, 34.3] |

### Paired differences at 3 mm (percentage points, bootstrap over the 200 shared episodes, success averaged over seeds)

| comparison (A minus B) | difference | 95% interval |
|---|---|---|
| mlp: ft minus noft | +9.2 | [+6.2, +12.5] |
| mlp_ft minus straight_to_estimate | +73.2 | [+67.0, +79.3] |
| mlp_noft minus straight_to_estimate | +64.0 | [+57.7, +70.3] |
| transformer: ft minus noft | +19.5 | [+15.3, +24.0] |
| transformer_ft minus straight_to_estimate | +72.8 | [+66.7, +79.0] |
| transformer_noft minus straight_to_estimate | +53.3 | [+46.7, +59.7] |
| diffusion: ft minus noft | +16.7 | [+13.2, +20.3] |
| diffusion_ft minus straight_to_estimate | +73.5 | [+67.5, +79.5] |
| diffusion_noft minus straight_to_estimate | +56.8 | [+50.8, +62.8] |
| diffusion_ft minus mlp_ft | +0.3 | [+0.0, +0.8] |
| diffusion_noft minus mlp_noft | -7.2 | [-10.7, -3.5] |
| diffusion_ft minus transformer_ft | +0.7 | [+0.2, +1.3] |
| diffusion_noft minus transformer_noft | +3.5 | [-0.2, +7.3] |
| transformer_ft minus mlp_ft | -0.3 | [-1.2, +0.5] |
| transformer_noft minus mlp_noft | -10.7 | [-14.5, -6.8] |

### Negative control: F/T-trained policies with the F/T reading blanked to zeros at test time (3 mm)

| policy | normal | F/T zeroed |
|---|---|---|
| MLP (BC) | 99.7% [98.8, 99.9] | 39.3% [35.5, 43.3] |
| Transformer chunk | 99.3% [98.3, 99.7] | 60.7% [56.7, 64.5] |
| Diffusion chunk | 100.0% [99.4, 100.0] | 67.2% [63.3, 70.8] |

### Diffusion policy: DDIM steps vs success (3 mm, PyTorch, 3 seeds x 100 episodes) and ONNX latency

| DDIM steps | with F/T success % | no F/T success % | ONNX 1-thread full loop median / p95 ms (best variant, seed 0) |
|---|---|---|---|
| 1 | 0.7% [0.2, 2.4] | 0.7% [0.2, 2.4] | 0.08 / 0.19 |
| 2 | 24.7% [20.1, 29.8] | 13.3% [9.9, 17.6] | 0.16 / 0.45 |
| 5 | 100.0% [98.7, 100.0] | 74.0% [68.8, 78.6] | 0.39 / 0.64 |
| 10 | 100.0% [98.7, 100.0] | 82.7% [78.0, 86.5] | 0.77 / 1.29 |
| 20 | 100.0% [98.7, 100.0] | 83.7% [79.1, 87.4] | 1.55 / 2.10 |

### On-board inference check (ONNX Runtime 1.32.0, CPU, batch 1, 500 calls, laptop not robot computer)

Control period at 20 Hz = 50 ms. Parity = max absolute difference ONNX vs PyTorch over 1,000 inputs (diffusion: the denoiser on 1,000 inputs and the full 10-step loop on 100). Timing was taken with load average 8.5 on 10 cores (other jobs were running).

| policy / threads | parity max abs diff | median ms | p95 ms | p95 < 50 ms | ONNX size KB |
|---|---|---|---|---|---|
| mlp (mlp_ft), threads 1 | 4.8e-07 | 0.006 | 0.017 | yes | 533 |
| mlp (mlp_ft), threads 10 | 4.8e-07 | 0.006 | 0.007 | yes | 533 |
| transformer (transformer_ft), threads 1 | 1.3e-06 | 0.088 | 0.226 | yes | 645 |
| transformer (transformer_ft), threads 10 | 1.3e-06 | 0.216 | 0.475 | yes | 645 |
| diffusion (diffusion_ft), 10 DDIM steps, threads 1 | 4.4e-05 | 0.77 | 1.29 | yes | 897 |
| diffusion (diffusion_ft), 10 DDIM steps, threads 10 | 4.4e-05 | 1.81 | 3.81 | yes | 897 |

### Training (CPU, 2 threads per process, 3 seeds trained in parallel on a shared machine)

| policy | parameters | epochs | mean wall seconds per model | mean val loss | mean train loss |
|---|---|---|---|---|---|
| MLP (BC), with F/T | 136195 | 100 | 21 | 0.0249 | 0.0074 |
| MLP (BC), no F/T | 134659 | 100 | 20 | 0.0233 | 0.0102 |
| Transformer chunk, with F/T | 152259 | 100 | 208 | 0.0287 | 0.0296 |
| Transformer chunk, no F/T | 151683 | 100 | 189 | 0.0288 | 0.0291 |
| Diffusion chunk, with F/T | 218051 | 200 | 355 | 0.1002 | 0.0954 |
| Diffusion chunk, no F/T | 217283 | 200 | 302 | 0.1014 | 0.0966 |

Losses are not comparable across rows (the MLP predicts 1 action, the others 8; diffusion's is a noise-prediction loss).

### Stage wall-clock times on this machine (`results/timings.json`)

| stage | seconds | load average after |
|---|---|---|
| train | 1142.1 | 47.1 |
| envcheck | 30.8 | 68.7 |
| demos | 4.2 | 72.7 |
| eval | 807.9 | 16.9 |
| ddim | 104.1 | 8.5 |
| onnx | 27.9 | 12.0 |
<!-- results:end -->
