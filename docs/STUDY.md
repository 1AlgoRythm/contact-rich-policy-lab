# Study guide: contact-rich-policy-lab

How to use this: Part 1 is one block per design decision (what I chose, the alternative, why, how I would know I was wrong). Part 2 is ten likely interview questions with answers drawn from this repo. Numbers inside answers are filled in from `results/*.json` by `scripts/render_readme.py` (they sit between `<!--n:key-->` markers); do not edit them by hand.

Plain-word glossary, first use:
- **Imitation learning / behaviour cloning (BC):** train a network to copy what an expert did, as a supervised regression from observation to action.
- **Action chunk:** a short sequence of future actions predicted at once.
- **Force/torque (F/T) sensor:** measures the push (3 numbers, newtons) and twist (3 numbers, newton-metres) passing through the robot's wrist.
- **Chamfer:** a sloped edge around a hole's mouth that funnels a peg in.
- **Wilson interval:** a 95% range for a success rate that stays sensible near 0% and 100% and for small counts.
- **Paired comparison:** two methods are tested on the same episodes, so the difference is not blurred by easy or hard episodes.
- **Privileged information:** something the expert knows (the true hole position) that the learner is never shown.
- **ONNX:** a file format for a trained network that other runtimes (here ONNX Runtime) can execute without PyTorch.

## Part 1. Design decisions

### 1. Simulator and physics: MuJoCo, a 3-axis sliding hand with spring-damper position control
- **Chose:** a hand that slides in x, y, z with a stiff position controller (2000 N/m), a peg welded to it, a block whose hole and chamfer are built from box geoms, gravity off.
- **Alternative:** a full robot arm model (7 joints), or a physics-lite 2D toy.
- **Why:** the question is about contact and force feedback, not kinematics. Three axes keep the policy output small and the simulation fast (thousands of control steps per second). A stiff position-controlled hand is how most real arms behave, and it is what makes wedging on a chamfer happen.
- **Wrong if:** the effect I report disappears when I change the controller stiffness or add rotation. I did not test that; it is a limit.

### 2. Action meaning: the setpoint integrates the action, with a 5 mm lead limit
- **Chose:** each action moves the position setpoint by up to 4 mm; the setpoint may not lead the real position by more than 5 mm.
- **Alternative:** "target = current position + action" (stateless).
- **Why:** my first version was stateless. The peg then had no lateral stiffness at all and slid into the chamfer on its own, so the baseline was far too good and force feedback was irrelevant. The persistent setpoint behaves like a real stiff arm. The 5 mm cap stops the setpoint running away while pressing (at most 10 N per axis).
- **Wrong if:** a different reasonable controller (impedance control) makes the baseline already good; then the "task needs force" premise is controller-specific. That is true, and I state it.

### 3. Force/torque sensor on a welded child body
- **Chose:** peg body welded to the hand, sensors on that weld. I read the MuJoCo docs: the sensor measures the interaction between the body holding the site and its parent, and the docs note it usually needs a dummy welded body.
- **Alternative:** sensor on the hand itself.
- **Why:** my first attempt put the site on the hand and the reading included the actuator's own drive force (about 0.9 N while merely accelerating in free air). That would let a policy read its own controller instead of contact. A test now pins this: in free flight the reading stays under 0.5 N while the actuators push over 1 N.
- **Wrong if:** the tests `test_ft_is_a_wrist_sensor_not_an_actuator_readout` or `test_ft_reads_the_pressing_force_hand_computed` fail.

### 4. Noisy hole estimate fixed per episode
- **Chose:** the policy sees the true hole position plus 3 mm Gaussian noise, drawn once per episode.
- **Alternative:** fresh noise every step (which the policy could average away).
- **Why:** a calibration or detection offset stays put in reality, so the policy cannot average it out. Only contact can correct it. That is what makes force feedback potentially useful.
- **Wrong if:** the no-F/T policies were as good as the F/T ones (then the noise is not the binding constraint).

### 5. Geometry chosen once, before training
- **Chose:** clearance 1 mm per side, chamfer 4 mm.
- **Alternative:** tune geometry until some result looks good.
- **Why:** I probed four geometries with the scripted baseline only; all gave a baseline success of 17 to 29 percent, i.e. neither trivial nor impossible, and I kept the first. The geometry sweep in `results/env_check.json` is reported in full.
- **Wrong if:** the conclusions flip at a different clearance; the sweep only covers the baseline, not the learned policies.

### 6. Expert: baseline approach, then privileged sliding after touchdown
- **Chose:** the expert approaches the noisy estimate exactly like the baseline (so a policy can copy it), and after the first contact (vertical force over 1 N) uses the true hole to slide in; random speeds and 0.1 action noise vary the demos. Only successes are kept; the expert's success over all attempts is reported.
- **Alternative:** an expert that is always perfectly aligned from the start (then demos contain no contact and F/T carries no information), or human demos.
- **Why:** the learner must see contact situations in the data to learn to react to force. The expert's privileged information is used only where it can be inferred from force by the learner.
- **Wrong if:** the expert's 100% success rate is mostly due to privilege the learner cannot reproduce. The F/T-zeroed control and the no-F/T variants measure how much of the gap closes.

### 7. Three policies, same inputs, same data, same loss family
- **Chose:** MLP predicting one action; transformer predicting 8 actions (one observation token plus 8 learned query tokens, no CVAE latent as in ACT); 1D conditional diffusion denoiser over an 8-step chunk, DDPM training with noise prediction, 10-step DDIM sampling, all written here. All use L2 loss, AdamW, EMA, standardised inputs.
- **Alternative:** use a diffusion library or the original ACT and Diffusion Policy code.
- **Why:** the interview question is "do you understand it", and about 60 lines of schedule and sampler are testable (closed-form noising test, oracle-noise sampler test). Sizes are matched loosely (135k to 220k parameters), not exactly.
- **Wrong if:** one policy loses only because I under-trained it. I used fixed epochs, no per-policy tuning, and I report the MLP's overfitting (train loss far below validation loss).

### 8. Chunk execution: predict 8, execute 4
- **Chose:** replan every 4 steps for the chunk policies, every step for the MLP.
- **Alternative:** execute the whole chunk, or all policies replan every step.
- **Why:** a middle value in the spirit of receding-horizon control; executing fewer actions than predicted keeps some reactivity to force. I did not sweep this.
- **Wrong if:** changing it to 1 or 8 reverses the ranking. Not tested; a limit.

### 9. Evaluation: shared seeds, pooling, intervals
- **Chose:** 200 episodes per seed with episode seeds shared by every variant; Wilson interval on the 600 pooled episodes plus mean and std over 3 seeds; paired bootstrap for differences.
- **Alternative:** independent random episodes per variant, mean over seeds only.
- **Why:** pairing removes episode-difficulty noise from comparisons. Pooled episodes are not independent (shared episodes), so I also give the seed spread.
- **Wrong if:** with more seeds the differences between the two chunk policies fall inside the seed spread (check the std column).

### 10. Negative controls
- **Chose:** (a) straight-to-estimate baseline, (b) blind expert in the tests, (c) F/T zeroed at test time, (d) no-F/T variants, (e) noise above the training range.
- **Why:** each could have falsified the main claim "force helps": if zeroing F/T did not hurt, or the no-F/T policy matched, the claim would be dead.
- **Wrong if:** the zeroed-F/T policy does as well as the normal one.

### 11. ONNX export and timing
- **Chose:** legacy `torch.onnx.export` (opset 17), the diffusion denoiser exported alone with the DDIM loop in numpy around ONNX Runtime; timing of the whole loop, batch 1, 500 calls, 1 thread and all threads.
- **Alternative:** export the sampler loop into the graph; quantise.
- **Why:** simplest thing that exercises the real cost: one network call per denoising step. Parity is checked on 1,000 inputs.
- **Wrong if:** parity differs by more than about 1e-4, or the machine load during timing was high (it was; see the recorded load average).

## Part 2. Ten likely interview questions

**Q1. What did you build, and what is the one-sentence finding?**
A MuJoCo peg-in-hole task with a wrist force/torque sensor, a scripted expert that makes demonstrations, and three small imitation-learning policies (MLP, transformer action chunks, diffusion action chunks), each trained with and without the F/T reading. Finding: in this simulation, giving the policy the F/T reading is what mattered (success at 3 mm hole-estimate noise rose by <!--n:gain_mlp-->+9.2 points (95% interval +6.2 to +12.5)<!--/n--> for the MLP, <!--n:gain_transformer-->+19.5 points (95% interval +15.3 to +24.0)<!--/n--> for the transformer and <!--n:gain_diffusion-->+16.7 points (95% interval +13.2 to +20.3)<!--/n--> for the diffusion policy); the choice among the three architectures mattered little once F/T was present at the training noise level, and only separated at higher noise.

**Q2. Why would force/torque help here, and how do you know it did?**
The hole position estimate is off by a few millimetres, which is more than the 1 mm clearance. Contact tells the peg which way the hole is: resting on the chamfer, the sideways force points toward the hole centre. Evidence in the repo: (1) the same policy trained without F/T is much worse (the baseline straight-to-estimate gets <!--n:base_3-->26.5% [20.9, 33.0]<!--/n--> and the no-F/T MLP <!--n:mlp_noft_3-->90.5% [87.9, 92.6]<!--/n--> versus <!--n:mlp_ft_3-->99.7% [98.8, 99.9]<!--/n--> with F/T); (2) the paired difference intervals above exclude zero; (3) the negative control: blanking the F/T numbers at test time drops the F/T-trained MLP from <!--n:mlp_ft_3-->99.7% [98.8, 99.9]<!--/n--> to <!--n:zero_mlp-->39.3% [35.5, 43.3]<!--/n-->, so the policy really uses them. Caveat I would volunteer: because the setpoint persists, the F/T also reveals how hard the controller is pushing, and I did not separate that from pure contact information.

**Q3. Did the diffusion policy beat the MLP and the transformer?**
Not clearly. With F/T at the training noise level all three are near the ceiling (diffusion <!--n:diffusion_ft_3-->100.0% [99.4, 100.0]<!--/n-->, MLP <!--n:mlp_ft_3-->99.7% [98.8, 99.9]<!--/n-->, transformer <!--n:transformer_ft_3-->99.3% [98.3, 99.7]<!--/n-->). Without F/T the MLP is best (MLP <!--n:mlp_noft_3-->90.5% [87.9, 92.6]<!--/n-->, transformer <!--n:transformer_noft_3-->79.8% [76.4, 82.8]<!--/n-->, diffusion <!--n:diffusion_noft_3-->83.3% [80.1, 86.1]<!--/n-->). The one place diffusion leads is the robustness probe with F/T at 8 mm noise (diffusion <!--n:diffusion_ft_8-->88.7% [85.9, 91.0]<!--/n-->, MLP <!--n:mlp_ft_8-->85.5% [82.5, 88.1]<!--/n-->, transformer <!--n:transformer_ft_8-->64.7% [60.8, 68.4]<!--/n-->). My reading, not tested: diffusion's known strength is representing several equally good behaviours, and my expert is nearly single-mode, so that strength is barely exercised, and with only about 10,000 training samples a plain MLP is hard to beat. Three seeds is a small sample, so I would not claim an ordering beyond what the intervals show.

**Q4. The expert knows the true hole position. Is that cheating?**
It is privileged on purpose, and the learner never sees it. Demonstrations in real life come from people who see more than the robot does, so this mimics that. To keep it honest: the expert approaches the noisy estimate exactly like the learner could, uses the true position only after touching the block, and only successful demos are kept (the expert succeeds <!--n:expert_rate-->100.0% [99.2, 100.0]<!--/n--> of the time). A test deliberately breaks it (give the expert the noisy estimate instead) and checks that its success collapses, which shows the privilege matters. The limit: a learner can only copy the parts of the expert's behaviour that its observations can explain.

**Q5. How did you make the comparison statistically fair?**
Fixed seeds; every controller is tested on the same 200 episodes per seed (same hole, same estimate error), 3 training seeds per variant; Wilson 95% intervals for rates; mean and std over seeds; paired bootstrap for differences. Weaknesses I would name: only 3 seeds; pooled episodes share the same 200 holes, so the pooled interval is somewhat optimistic; and ceilings (near 100%) hide differences.

**Q6. Explain how your diffusion policy is trained and sampled, and how you tested it.**
Training: take a true 8-step action chunk, pick a random noise level t, mix: noisy = sqrt(abar_t) times the chunk plus sqrt(1 minus abar_t) times Gaussian noise (abar_t is the running product of "how much signal is kept" per step). The network gets the noisy chunk, t, and the observation, and learns to predict the noise (epsilon-prediction, squared error). Sampling: start from pure noise; at each of 10 DDIM steps predict the noise, back out an estimate of the clean chunk (clipped to [-1, 1]), and re-noise that estimate to the next lower noise level using the same predicted noise. Tests: the forward noising matches the closed-form mean sqrt(abar) times x0 and variance 1 minus abar over 200,000 samples; and if the network were perfect (returned the true noise) the sampler returns the clean chunk exactly for 1, 5, 10 or 50 steps. Schedule formula and clipping are from memory, not checked against the papers.

**Q7. What is action chunking, and why predict 8 and execute 4?**
Predict several future actions at once instead of one: smoother motion and fewer compounding one-step errors (ACT, abstract read). Executing only the first 4 and replanning keeps the policy able to react to contact every 200 ms. I did not sweep the numbers; that is untested, and an open-loop chunk is exactly where force reactivity could suffer.

**Q8. Could it run on a robot in real time?**
On this laptop CPU, with one thread, the MLP takes about <!--n:onnx_mlp-->0.006 / 0.017<!--/n--> ms per call (median / 95th percentile), the transformer <!--n:onnx_transformer-->0.088 / 0.226<!--/n--> ms, and the diffusion policy's 10-step loop <!--n:onnx_diffusion10-->0.77 / 1.29<!--/n--> ms, against a 50 ms budget at 20 Hz; the diffusion cost grows about linearly with steps (1 step <!--n:onnx_diffusion1-->0.08 / 0.19<!--/n--> ms, 50 steps <!--n:onnx_diffusion50-->4.30 / 5.04<!--/n--> ms). The machine was shared with other jobs, so these are rough, and it is not a robot computer. ONNX outputs match PyTorch to within about <!--n:parity_max-->4.4e-05<!--/n--> (largest max absolute difference across the three exports).

**Q9. What went wrong while building it, and what do you distrust?**
Three things I fixed: (1) my first F/T sensor sat on the hand and read the actuator's own force, so I moved it to a welded peg body and added a test; (2) my first action definition had no lateral stiffness, so the peg slid in on its own and force feedback was irrelevant, so I switched to a persistent setpoint; (3) my first chamfer slabs were rotated and placed wrongly and blocked the hole, which showed up as 0% success even for the expert. Things I still distrust: the setpoint leak into F/T (Q2), the MLP overfitting (training loss far below validation loss), only 3 seeds, and timing on a shared machine.

**Q10. What would you do with a real robot?**
Start by measuring, not training: log the real F/T sensor at rest and in contact to get its noise, bias drift and delay, and tare it before every run. Replace scripted demos with teleoperated ones (they will be multimodal, which is where a diffusion policy can finally earn its place), and build the data pipeline first: synchronise timestamps, drop failed or sensor-dropout episodes, check force ranges, keep a held-out set. Add hard force limits and a safe stop before running any learned policy. Randomise friction, clearance, stiffness and sensor noise in simulation, and compare against a plain impedance-control insertion baseline, which is the thing a real team would already have. Evaluate on a fixed protocol with enough trials for an interval. I would not expect these simulated numbers to transfer.
