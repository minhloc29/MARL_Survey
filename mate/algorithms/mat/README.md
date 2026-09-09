# MAT integration

This package contains a MATE-native autoregressive Multi-Agent Transformer
trainer. It ports the useful MAT ideas without importing the original
repository's legacy Gym and PyTorch dependency stack.

Run a short experiment from the MATE project root:

```bash
PYTHONPATH=. uv run python train.py \
  --algo mat \
  --mat-config MATE-4v8-9.yaml \
  --mat-levels 3 \
  --mat-episode-limit 64 \
  --mat-updates 1000 \
  --mat-rollout-steps 128 \
  --mat-ppo-epochs 4 \
  --mat-minibatch-size 64 \
  --device cpu
```

Use `--device mps` on Apple Silicon or `--device cuda` on a CUDA machine.
The checkpoint is saved to `ckpts/mat/latest.pt` by default.

The trainer uses `MultiCamera` with `GreedyTargetAgent`, discrete camera-grid
actions, an attention encoder over all cameras, a causal autoregressive decoder
that selects camera actions in order, centralized value prediction from the
joint state, GAE, and clipped PPO updates.

This first integration targets cooperative camera control against a fixed
target policy. It does not yet provide continuous MAT actions, self-play, or
parallel rollout workers.