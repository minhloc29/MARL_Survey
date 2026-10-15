# IQL integration

This package contains a MATE-native implementation of Independent Q-Learning
(IQL) for discrete cooperative camera control. In IQL, each agent runs its own
DQN over its own observations only, treating the environment as single-agent and
ignoring teammates.

Run a short smoke test from the MATE project root:

```bash
PYTHONPATH=. uv run python train.py \
  --algo iql \
  --iql-config MATE-1v2-0.yaml \
  --iql-levels 3 \
  --iql-episode-limit 64 \
  --iql-episodes 1000 \
  --iql-update-every 8 \
  --iql-batch-size 8 \
  --device cpu
```

Use `--device mps` on Apple Silicon or `--device cuda` on a CUDA machine.
The checkpoint is saved to `ckpts/iql/latest.pt` by default.

The implementation uses:

- `MultiCamera` with `GreedyTargetAgent` as the fixed opponent;
- the discrete camera grid from the RODE environment adapter;
- a shared feed-forward Q-network with camera identity embeddings;
- independent per-agent epsilon-greedy exploration;
- target-network TD bootstrapping (`r + gamma * max_a' Q_target(o', a')`);
- episode replay and masked variable-length updates.

There is no mixer and no shared state: each agent maximizes its own expected
return independently, which makes IQL a simple but often unstable cooperative
baseline (agents do not coordinate). This is intentionally a camera-team
baseline; continuous-action and co-trained variants are out of scope.
