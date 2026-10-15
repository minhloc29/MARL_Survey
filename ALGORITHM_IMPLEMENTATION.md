# Implementing MARL algorithms faithfully

A practical field-guide for adding a new algorithm to `mate/algorithms/` (or any
multi-agent RL project) such that it actually trains and matches the intent of
the original paper. It is written from the concrete failure modes that were
found and fixed in this codebase, so the "don't" list is not theoretical — each
one broke a real learner here.

The canonical references for MA algorithms are the original papers plus the
PyMARL and EPyMARL reference codebases. This note is the glue between "paper"
and "a numpy+torch implementation that converges."

---

## 0. The single most common silent killer: input scaling

Raw sensor/coordinate features are almost never in the range your network
assumes. In this environment the observations/state are raw coordinates ranging
from about **−1000 to +1500** (std ~350). Feeding that into a plain
`nn.Linear` inflates the first-layer activations, which then:

- **For a policy head** pushes logits to extreme magnitudes → softmax becomes
  near one-hot → **entropy collapses to ~0 within a rollout**, the policy
  becomes deterministic, gradients vanish, and the reported policy loss pins to
  `0.0000`.
- **For a Tanh head / GRU / Q-head** saturates the activation → vanishing
  gradients and a value/critic that cannot fit its targets.

**Rule:** normalize inputs at the boundary before any network sees them. In this
repo the fix lives once in `mate/algorithms/rode/environment.py` (`_normalize`),
standardizing every feature to zero-mean unit-variance per sample (feature axis)
inside `reset()` and `step()`. Centralizing it at the env wrapper means every
algorithm inherits it and no network needs its own LayerNorm.

Checklist:

- [ ] Print the raw min/max/std of your obs and state before training.
- [ ] If the spread exceeds ~`[−10, 10]`, normalize (per-sample standardization,
      `LayerNorm`, or a fixed running statistics normalizer).
- [ ] Confirm policy entropy at **initialization** on *real* obs is near the
      analytic max (`ln(n_actions)`). If it is ~0 before any training, you have
      a scaling (or logit) bug, not a learning problem.

---

## 1. Nail down the tensor layout first, then the math

The most confusing bugs (the COMA crash in this repo, `RuntimeError: size of
tensor a (64) must match tensor b (4)`) come from assuming a shape that the data
does not have. Decide the layout once and write it down:

Common choices for multi-agent data:

- **`(batch, time, agents, feature)`** — used by the sequence/transformer
  learners here (obs stacking joints and steps).
- **`(batch, agents, feature)`** — per-step batch (no sequence).
- **`(batch, time, feature)`** with an explicit scalar **agents** dim — flat
  state / replay sampler shapes.

The two failure classes are:

**Wrong dim ordering.** In `coma.py`, `q_values` came back as
`(batch, steps, agents, n_actions)` (it is per-agent), but the target expression
did `next_q_taken.unsqueeze(-1)`, producing `(batch, steps, agents, 1)`, and
then broadcast `(batch, steps, 1) * (batch, steps, agents, 1)`. Right-aligned
broadcasting matched **steps** against **agents**, so `64` hit `4` and crashed.
The fix was to *not* unsqueeze, letting `(batch, steps, 1) * (batch, steps,
agents)` broadcast to per-agent targets.

**Wrong axis.** `rewards`/`terminated`/`mask` often carry a trailing `1`
(`(B, T, 1)`), while per-agent quantities don't. Constantly verify which side
has the agent dimension and which has the singleton.

Checklist:

- [ ] Write each tensor's shape in a comment at the top of `update()`.
- [ ] When broadcasting `(…, 1)` against `(…, K)`, confirm the `1` is *right-aligned
      to the intended singleton*, i.e. you are not accidentally aligning `T` with
      `A`.
- [ ] Add a tiny `assert` or quick unit shape check the first time each method
      runs.
- [ ] Verify with real-shaped data before training — not synthetic random
      tensors, which hide layout mismatches.

---

## 2. Faithful PPO (on-policy instinct)

`MA`-PPO (MAPPO/COMA/MAT/IPPO) all rest on the same core. The things people
implement wrong:

**GAE.**
```
delta = r + gamma * V(next) * (1 - done) - V
adv    = delta + gamma * lambda * (1 - done) * adv
```
- `done` must be the *environment* terminal, **not** the time/truncation limit,
  unless you explicitly choose to truncate value bootstrapping. Bugs here
  silently corrupt every target.
- `V(next)` for the last step must be a real bootstrap value, not zero-filled —
  collect a final value via a no-grad forward pass.

**Normalize advantages** (per minibatch or per rollout) — `(adv - mean) /
(std + eps)`. This is standard practice and makes the policy loss scale sane.
Skipping it does not crash, but degrades stability and makes reported losses
look huge.

**Ratio clipping.**
```
ratio  = exp(log pi_new(a) - log pi_old(a))
clipped= clamp(ratio, 1-eps, 1+eps)
loss   = -min(ratio * A, clipped * A).mean()
```
Use the *joint* action's total log-prob (sum over agents for fully-cooperative
joint policies, or per-agent for each head) consistently on both the stored
`old_log_prob` and the recomputed `new_log_prob`. A mismatch (e.g. computing one
per-agent and the other joint) silently wrecks the ratio.

**`old_log_prob` must be stored from the same distribution that acted**, under
`no_grad`, at collection time — not recomputed from the current (already
updated) policy.

**Entropy bonus:** add `beta * H(pi)` with a small `beta` (e.g. 0.01). Too weak
and the policy collapses to determinism (see §0). Print entropy over training —
it should not drop to 0 instantly unless you intend it.

Checklist:

- [ ] `old_log_prob` stored at collection under no_grad, same dist as acting.
- [ ] Advantage/target uses the correct (env-terminal) done mask.
- [ ] Last-step bootstrap value collected, not zeroed.
- [ ] Advantages normalized.
- [ ] Entropy printed and monitored.

---

## 3. Faithful Q-learning (off-policy instinct)

COMA/DCG/I2C/FACMAC use temporal-difference targets. The classic bugs:

**Target network.** Use a separate `target_*` network, hard-copied periodically
(`target_update_every`) — never train directly on the live net's Q of the next
state. Many bugs come from forgetting `target_net.load_state_dict(net.state_dict())`
under the right schedule.

**Target uses `gamma * max_a Q_target(s', a)`** — for policies with a target net,
the *target* network computes the max/expected Q over the next state. If you
gather the next action from the *live* policy but the Q value from the target
net, be explicit about which is which.

**Batch vs episode.** Off-policy learners sample transitions from a replay
buffer. Sequences (agents + steps) need proper padding (`filled`/`mask`), and
the loss must be masked by the `mask` so padded zeros don't count as real
transitions. In this repo the replay returns `mask`/`filled`; forgetting to
apply it inflates the loss with garbage.

Checklist:

- [ ] Target network exists, is a `deepcopy`, and is hard-updated.
- [ ] TD target uses target-net Q on next state.
- [ ] Replay-sample losses are masked by `filled`/`mask`.
- [ ] Done/terminated handled consistently in the target.

---

## 4. Autoregressive / transformer heads (MAT-style)

For sequence-decision transformers:

- The decoder must be **causal** (`torch.triu(mask, diagonal=1)`) so agent *i*
  conditions on agents `0..i-1` only.
- During **sampling**, fill actions agent-by-agent and re-forward: the standard
  greedy autoregressive loop (`actions` starts as a padding token, each agent
  sampled and written back, final pass for log-probs/values).
- During **training**, pass the recorded actions with a shifted previous-action
  embedding (padding token + `actions[..., :-1]`).
- **Important:** the *sampled* action used for `old_log_prob` and the one used
  for the joint-marginal must be the same sequence, otherwise the PPO ratio in
  §2 is inconsistent.

---

## 5. Verify a whole training run, not just forward shapes

A shape fix that makes the loss "not crash" is not the same as training
correctly. Minimal sanity a short run should show:

1. **Initial entropy ≈ ln(n_actions)** at update 1 (see §0).
2. **Policy loss is non-trivial** (not pinned at exactly `0.0000` across many
   updates) — if it is, the ratio is fully clipped or gradients vanished.
3. **Value/critic loss is learning** (moving, ideally decreasing as the value
   head fits bootstrapped returns) rather than saturating at a huge constant.
4. The reward is at least in the neighbourhood of a heuristic/random baseline —
   it need not solve the task quickly, but it should not be a flat degenerate
   curve from the start.

If any of these are broken, suspect, in order: input scaling (§0), tensor
layout (§1), then target/ratio details (§2–§3).

---

## 6. Faithfulness checklist (before you call it "implemented")

- [ ] The loss matches the paper (write the paper's equation, then the code,
      side by side).
- [ ] Shapes are explicit and verified on real data.
- [ ] Inputs are normalized.
- [ ] Targets/advantages use the right done/filled masks and target networks.
- [ ] `old_log_prob` / `log_prob`/`ratio` are internally consistent.
- [ ] Entropy, policy loss, value loss, and reward all printed and sane over a
      real short run.
- [ ] Configuration (learning rates, batch sizes, horizons) matches the paper or
      the reference implementation where it matters for convergence.
- [ ] You can point to the exact lines implementing each term of the algorithm.

---

*See also:* the PyMARL codebase for a canonical multi-agent implementation, and
each algorithm's own `mate/algorithms/<name>/README.md` for the runnable
example in this repo.
