# COMA integration

This package contains a MATE-native implementation of Counterfactual Multi-Agent
Policy Gradients for discrete cooperative camera control.

Run a short smoke test from the MATE project root:

```bash
PYTHONPATH=. uv run python train.py \
  --algo coma \
  --coma-config MATE-1v2-0.yaml \
  --coma-levels 3 \
  --coma-episode-limit 64 \
  --coma-episodes 1000 \
  --coma-update-every 8 \
  --coma-batch-size 8 \
  --device cpu
```

Use `--device mps` on Apple Silicon or `--device cuda` on a CUDA machine.
The checkpoint is saved to `ckpts/coma/latest.pt` by default.

The implementation uses:

- `MultiCamera` with `GreedyTargetAgent` as the fixed opponent;
- the discrete camera grid from the RODE environment adapter;
- a shared categorical actor with camera identity embeddings;
- a centralized critic over the flattened joint state and one-hot actions;
- counterfactual action baselines for each camera;
- target-critic TD bootstrapping;
- episode replay and masked variable-length updates.

This is intentionally a camera-team baseline. Continuous-action COMA and
co-training both teams require separate action distributions and environment
coordination layers.