"""Numerical, sequence-boundary and real-environment checks for MAMBA."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mate.algorithms.mamba import MAMBAConfig, MAMBATrainer
from mate.algorithms.mamba.core.critic import AugmentedCritic
from mate.algorithms.mamba.core.loss_utils import compute_return
from mate.algorithms.mamba.environment import MAMBAEnvironment
from mate.algorithms.mamba.replay import SequenceReplay


class MAMBATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_time_alignment_and_truncation(self):
        replay = SequenceReplay(4, 3, 4)
        obs = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
        replay.add(obs, [[1], [2], [3]], [10, 20, 30], [False, False, False])
        batch = replay.sample(1)
        np.testing.assert_array_equal(batch["observation"][:, 0, 0, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(batch["action"][1:, 0, 0].argmax(-1), [1, 2, 3])
        self.assertEqual(batch["action"][0].sum(), 0)
        np.testing.assert_array_equal(batch["reward"][:, 0, 0, 0], [0, 10, 20, 30])
        self.assertEqual(batch["done"].sum(), 0)  # time limit can bootstrap
        self.assertEqual(batch["last"][-1].item(), 1)  # still a recurrent boundary

    def test_terminal_target(self):
        replay = SequenceReplay(2, 2, 2)
        replay.add(np.zeros((3, 1, 1)), [[0], [1]], [0, 1], [False, True])
        np.testing.assert_array_equal(replay.sample(1)["done"][:, 0, 0, 0], [0, 0, 1])

    def test_sequences_never_cross_episodes_and_capacity_is_bounded(self):
        replay = SequenceReplay(8, 2, 2)
        for marker in (1, 2):
            replay.add(np.full((5, 2, 1), marker), np.zeros((4, 2)), np.zeros(4), np.zeros(4))
        batch = replay.sample(64)["observation"]
        self.assertTrue(torch.all(batch == batch[:1]))
        replay.add(np.full((5, 2, 1), 3), np.zeros((4, 2)), np.zeros(4), np.zeros(4))
        self.assertEqual(len(replay), 8)
        self.assertTrue(torch.all(replay.sample(64)["observation"] >= 2))

    def test_short_episode_is_not_sampled(self):
        replay = SequenceReplay(8, 3, 2)
        replay.add(np.zeros((2, 1, 1)), [[0]], [0], [True])
        self.assertFalse(replay.ready)
        with self.assertRaises(ValueError):
            replay.sample(1)

    def test_invalid_config_and_terminal_boundary(self):
        for kwargs in ({"hidden": 30}, {"sequence_length": 1}, {"horizon": 1},
                       {"episode_limit": 2}, {"warmup": 100001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MAMBAConfig(**kwargs)
        replay = SequenceReplay(4, 2, 2)
        with self.assertRaises(ValueError):
            replay.add(np.zeros((3, 1, 1)), [[0], [1]], [0, 0], [True, False])

    def test_lambda_returns_use_continuation(self):
        reward = torch.tensor([1., 2.]).reshape(2, 1, 1, 1)
        value = torch.zeros_like(reward)
        discount = torch.tensor([1., 0.]).reshape_as(reward)
        result = compute_return(reward, value, discount, torch.full((1, 1, 1), 99.), 1., 0.5)
        torch.testing.assert_close(result.flatten(), torch.tensor([2., 2.]))

    def test_critic_preserves_time_batch_and_agent_axes(self):
        critic = AugmentedCritic(12, 32)
        for agents in (1, 4, 31):
            with self.subTest(agents=agents):
                value = critic(torch.randn(3, 2, agents, 12))
                self.assertEqual(value.shape, (3, 2, agents, 1))

    def test_real_mate_discrete_actions_and_time_limit(self):
        env = MAMBAEnvironment(levels=3, episode_limit=2)
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (4, env.obs_dim))
            self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(env.n_actions, 9)
            with self.assertRaises(ValueError):
                env.step(np.full(4, 9))
            env.step(np.full(4, 4))
            obs, _, _, truncated, _ = env.step(np.full(4, 4))
            self.assertTrue(truncated)
            self.assertTrue(np.isfinite(obs).all())
        finally:
            env.close()

    def test_real_updates_and_checkpoint_roundtrip(self):
        config = MAMBAConfig(episodes=1, episode_limit=4, sequence_length=2,
                             batch_size=2, hidden=32, categoricals=2, classes=4,
                             horizon=3, ppo_epochs=2, warmup=0, wandb_api="test-secret")
        trainer = MAMBATrainer(config)
        try:
            before = {name: torch.cat([p.detach().flatten().clone() for p in getattr(trainer, name).parameters()])
                      for name in ("model", "actor", "critic")}
            trainer.collect_episode(0)
            for _ in range(2):
                metrics = trainer.update()
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            for name, initial in before.items():
                current = torch.cat([p.detach().flatten() for p in getattr(trainer, name).parameters()])
                self.assertFalse(torch.equal(initial, current), name)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "latest.pt"
                trainer.save(path)
                payload = torch.load(path, weights_only=True)
                self.assertNotIn("wandb_api", payload["config"])
                restored = MAMBATrainer.from_checkpoint(path)
                try:
                    self.assertEqual(restored.updates, 2)
                    self.assertEqual(len(restored.replay), 0)
                    for name in ("model", "actor", "critic", "target_critic"):
                        for key, tensor in getattr(trainer, name).state_dict().items():
                            torch.testing.assert_close(tensor, getattr(restored, name).state_dict()[key], rtol=0, atol=0)
                    obs, _ = trainer.env.reset(seed=123)
                    torch.manual_seed(123)
                    expected, _, _ = trainer.act(obs, greedy=True)
                    torch.manual_seed(123)
                    actual, _, _ = restored.act(obs, greedy=True)
                    np.testing.assert_array_equal(actual, expected)
                    restored.collect_episode(1)
                    self.assertTrue(all(np.isfinite(v) for v in restored.update().values()))
                finally:
                    restored.env.close()
        finally:
            trainer.env.close()


if __name__ == "__main__":
    unittest.main()
