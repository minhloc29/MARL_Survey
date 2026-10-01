"""Algorithmic and real-MATE checks for the AORPO integration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mate.algorithms.aorpo import AORPOConfig, AORPOTrainer
from mate.algorithms.aorpo.aorpo import bellman_target
from mate.algorithms.aorpo.networks import DynamicsEnsemble, adaptive_lengths, discrete_action
from mate.algorithms.aorpo.replay import TransitionReplay, mixed_batch


def small_config(**kwargs):
    values = dict(episodes=1, episode_limit=8, hidden=32, model_hidden=32,
                  ensemble_size=2, batch_size=4, model_batch_size=4, model_samples=8,
                  model_epochs=1, opponent_epochs=1, policy_updates=2, warmup=4,
                  rollout_batch_size=4, rollout_min=3, rollout_horizon=3)
    return AORPOConfig(**dict(values, **kwargs))


def transition(marker, n_agents=2):
    obs = np.full((n_agents, 3), marker, dtype=np.float32)
    return obs, np.zeros(n_agents, dtype=np.int64), np.full(n_agents, marker), obs + 1, False


class AORPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_adaptive_lengths(self):
        torch.testing.assert_close(adaptive_lengths(torch.tensor([1., .5, 0.]), 4), torch.tensor([4, 2, 0]))
        torch.testing.assert_close(adaptive_lengths(torch.tensor([.6, .3]), 4), torch.tensor([4, 2]))
        self.assertEqual(adaptive_lengths(torch.zeros(3), 4).sum(), 0)
        self.assertEqual(adaptive_lengths(torch.empty(0), 4).numel(), 0)
        with self.assertRaises(ValueError):
            adaptive_lengths(torch.tensor([float("nan")]), 4)

    def test_replay_capacity_and_mixing(self):
        real, model = TransitionReplay(3), TransitionReplay(2)
        for i in range(5):
            real.add(*transition(i))
        np.testing.assert_array_equal(real.latest(3)["rewards"][:, 0], [2, 3, 4])
        self.assertEqual(len(real), 3)
        model.add(*transition(99))
        batch = mixed_batch(real, model, 8, 0.25)
        self.assertEqual((batch["rewards"][:, 0] == 99).sum(), 6)
        self.assertEqual(mixed_batch(real, model, 3, 1)["obs"].shape[0], 3)
        self.assertTrue((mixed_batch(real, model, 3, 0)["rewards"] == 99).all())
        model.clear()
        self.assertTrue((mixed_batch(real, model, 3, 0)["rewards"] != 99).all())
        with self.assertRaises(ValueError):
            model.sample(1)

    def test_bellman_terminal_and_truncation(self):
        result = bellman_target(torch.tensor([2., 2.]), torch.tensor([1., 0.]), torch.tensor([10., 10.]), .9)
        torch.testing.assert_close(result, torch.tensor([2., 11.]))

    def test_categorical_log_prob_and_gradient(self):
        logits = torch.tensor([[1., 2., 3.]], requires_grad=True)
        action, log_prob = discrete_action(logits, stochastic=True)
        self.assertEqual(action.sum(), 1)
        expected = logits.log_softmax(-1).gather(-1, action.argmax(-1, keepdim=True)).squeeze(-1)
        torch.testing.assert_close(log_prob, expected)
        (-log_prob.mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(), 0)

    def test_ensemble_single_sample_normalization_and_variance(self):
        model = DynamicsEnsemble(1, 3, 2, hidden=8, members=2)
        obs, actions = torch.ones(1, 1, 3), torch.zeros(1, 1, dtype=torch.long)
        inputs = model.inputs(obs, actions)
        model.fit_stats(inputs)
        self.assertTrue(torch.isfinite(model.input_std).all())
        self.assertTrue((model.input_std > 0).all())
        mean, logvar, _ = model(inputs)
        self.assertEqual(mean.shape, (2, 1, 4))
        self.assertTrue((logvar >= model.min_logvar - 1e-6).all())
        with patch("torch.randn_like", side_effect=lambda x: torch.ones_like(x)):
            # Make both ensemble members agree; noise=1 yields exactly mean+std.
            with patch.object(model, "forward", return_value=(torch.zeros(2, 1, 4),
                              torch.full((2, 1, 4), float(np.log(4.))), torch.full((2, 1), -100.))):
                nxt, reward, terminated = model.predict(obs, actions)
                torch.testing.assert_close(nxt, torch.full_like(nxt, 2.))
                torch.testing.assert_close(reward, torch.full_like(reward, 2.))
                self.assertEqual(terminated.item(), 0)

    def test_opponent_wise_switch_and_query_counts(self):
        trainer = AORPOTrainer(small_config())
        try:
            ego = trainer.agents[0]
            ego.match_rates.copy_(torch.tensor([1., .5, 0.]))
            obs = torch.zeros(2, 4, trainer.env.obs_dim)
            def prediction(states, actions):
                return states + .01, torch.zeros(len(states), 4), torch.zeros(len(states))
            with patch.object(ego.dynamics, "predict", side_effect=prediction), \
                 patch.object(trainer.env, "step", side_effect=AssertionError("Model rollouts must not step MATE")), \
                 patch.object(trainer.agents[1], "policy", wraps=trainer.agents[1].policy) as p1, \
                 patch.object(trainer.agents[2], "policy", wraps=trainer.agents[2].policy) as p2, \
                 patch.object(trainer.agents[3], "policy", wraps=trainer.agents[3].policy) as p3:
                self.assertEqual(trainer.rollout_agent(0, obs, horizon=4), 8)
                self.assertEqual((p1.call_count, p2.call_count, p3.call_count), (0, 2, 4))
            torch.testing.assert_close(ego.query_counts, torch.tensor([0, 4, 8]))
            batch = trainer.model_replays[0].latest(8)
            torch.testing.assert_close(batch["next_obs"], batch["obs"] + .01)
        finally:
            trainer.env.close()

    def test_model_rollout_stops_at_predicted_terminal(self):
        trainer = AORPOTrainer(small_config())
        try:
            obs = torch.zeros(3, 4, trainer.env.obs_dim)
            with patch.object(trainer.agents[0].dynamics, "predict", return_value=(obs, torch.ones(3, 4), torch.ones(3))):
                self.assertEqual(trainer.rollout_agent(0, obs, 5), 3)
            self.assertEqual(len(trainer.model_replays[0]), 3)
            self.assertTrue(trainer.model_replays[0].latest(3)["terminated"].all())
        finally:
            trainer.env.close()

    def test_schedule_and_invalid_configuration(self):
        trainer = AORPOTrainer(small_config(rollout_min=1, rollout_horizon=5, rollout_start=10, rollout_end=20))
        try:
            for episode, expected in [(0, 1), (10, 1), (15, 3), (20, 5), (30, 5)]:
                trainer.episode = episode
                self.assertEqual(trainer.horizon, expected)
        finally:
            trainer.env.close()
        for kwargs in ({"ensemble_size": 0}, {"real_ratio": 2}, {"rollout_min": 9}, {"replay_capacity": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                small_config(**kwargs)

    def test_real_mate_updates_and_checkpoint_resume(self):
        trainer = AORPOTrainer(small_config(wandb_api="secret-not-for-checkpoints"))
        try:
            before = {name: torch.cat([p.detach().flatten().clone() for p in getattr(trainer.agents[0], name).parameters()])
                      for name in ("actor", "critics", "opponents", "dynamics")}
            rollout = trainer.collect_episode(0)
            self.assertEqual(rollout["length"], 8)
            self.assertEqual(len(trainer.model_train), 6)
            self.assertEqual(len(trainer.model_validation), 2)
            self.assertEqual(trainer.real_replay.latest(8)["terminated"].sum(), 0)
            for _ in range(2):
                self.assertTrue(all(np.isfinite(v) for v in trainer.update().values()))
            for name, original in before.items():
                current = torch.cat([p.detach().flatten() for p in getattr(trainer.agents[0], name).parameters()])
                self.assertFalse(torch.equal(original, current), name)
            self.assertNotEqual(trainer.agents[0].log_alpha.item(), 0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "latest.pt"
                trainer.save(path)
                self.assertNotIn("wandb_api", torch.load(path, weights_only=True)["config"])
                restored = AORPOTrainer.from_checkpoint(path)
                try:
                    self.assertEqual(restored.updates, 2)
                    self.assertEqual(len(restored.real_replay), 0)
                    for name, tensor in trainer.agents.state_dict().items():
                        torch.testing.assert_close(tensor, restored.agents.state_dict()[name], rtol=0, atol=0)
                    obs, _ = trainer.env.reset(seed=20)
                    np.testing.assert_array_equal(trainer.act(obs, greedy=True), restored.act(obs, greedy=True))
                    # Alpha remains the optimizer's registered leaf after loading.
                    for agent, opts in zip(restored.agents, restored.optimizers):
                        self.assertIs(opts["alpha"].param_groups[0]["params"][0], agent.log_alpha)
                    restored.collect_episode(1)
                    alpha = restored.agents[0].log_alpha.detach().clone()
                    self.assertTrue(all(np.isfinite(v) for v in restored.update().values()))
                    self.assertFalse(torch.equal(alpha, restored.agents[0].log_alpha))
                finally:
                    restored.env.close()
        finally:
            trainer.env.close()

    def test_single_camera_has_no_opponent_reduction_errors(self):
        trainer = AORPOTrainer(small_config(config="MATE-1v2-0.yaml"))
        try:
            trainer.collect_episode(0)
            self.assertTrue(all(np.isfinite(v) for v in trainer.update().values()))
            self.assertEqual(trainer.agents[0].query_counts.numel(), 0)
        finally:
            trainer.env.close()


if __name__ == "__main__":
    unittest.main()
