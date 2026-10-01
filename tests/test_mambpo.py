"""MAMBPO continuous-action, decentralized-policy and world-model checks."""

from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mate.algorithms.camera_environment import ContinuousCameraEnvironment
from mate.algorithms.mambpo import MAMBPOConfig, MAMBPOTrainer
from mate.algorithms.mambpo.networks import GaussianActor, WorldModel
from mate.algorithms.model_based_replay import TransitionReplay, mixed_batch


def small_config(**kwargs):
    config = dict(episodes=1, episode_limit=8, actor_hidden=32, critic_hidden=32,
                  model_hidden=32, ensemble_size=2, batch_size=4, model_batch_size=4,
                  model_steps=1, policy_updates=2, warmup=4, rollout_batch_size=4, update_every=4)
    return MAMBPOConfig(**dict(config, **kwargs))


class MAMBPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_continuous_environment_scaling_and_time_limit(self):
        env = ContinuousCameraEnvironment(episode_limit=2)
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (4, env.obs_dim))
            actions = np.zeros((4, env.action_dim))
            np.testing.assert_array_equal(env.to_environment_actions(actions - 1), env.action_low)
            np.testing.assert_array_equal(env.to_environment_actions(actions + 1), env.action_high)
            np.testing.assert_allclose(env.to_environment_actions(actions), (env.action_low + env.action_high) / 2)
            with self.assertRaises(ValueError):
                env.step(actions + 2)
            with self.assertRaises(ValueError):
                env.step(np.full_like(actions, np.nan))
            env.step(actions)
            obs, _, terminated, truncated, _ = env.step(actions)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(np.isfinite(obs).all())
        finally:
            env.close()

    def test_replay_preserves_fractional_actions_and_mixture(self):
        real = TransitionReplay(2, continuous_actions=True)
        model = TransitionReplay(2, continuous_actions=True)
        obs = np.zeros((2, 3), dtype=np.float32)
        action = np.array([[-.25, .5], [.1, -.8]], dtype=np.float32)
        real.add(obs, action, np.zeros(2), obs + 1, False)
        model.add(obs + 8, -action, np.ones(2), obs + 9, True)
        batch = mixed_batch(real, model, 4, .5)
        torch.testing.assert_close(batch["actions"][0], torch.from_numpy(action))
        torch.testing.assert_close(batch["actions"][2], -torch.from_numpy(action))
        self.assertEqual(batch["terminated"].sum(), 2)
        discrete = TransitionReplay(2)
        with self.assertRaises(ValueError):
            discrete.add(obs, action, np.zeros(2), obs, False)

    def test_gaussian_actor_tanh_jacobian_and_gradient(self):
        actor = GaussianActor(3, 2, hidden=8)
        with torch.no_grad():
            for p in actor.parameters():
                p.zero_()
        obs = torch.zeros(4, 3)
        action, log_prob = actor(obs, greedy=True)
        torch.testing.assert_close(action, torch.zeros_like(action))
        torch.testing.assert_close(log_prob, torch.full((4,), -float(np.log(2 * np.pi))))
        action, log_prob = actor(obs)
        self.assertTrue((action.abs() <= 1).all())
        (-log_prob.mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters()))
        with torch.no_grad():
            actor.net[-1].bias[:2].fill_(30.)
        _, saturated_log_prob = actor(obs, greedy=True)
        self.assertTrue(torch.isfinite(saturated_log_prob).all())

    def test_actor_uses_only_own_observation(self):
        trainer = MAMBPOTrainer(small_config())
        try:
            obs, _ = trainer.env.reset(seed=0)
            original = trainer.act(obs, greedy=True)
            for camera in range(4):
                changed = obs.copy()
                changed[np.arange(4) != camera] += 100.
                np.testing.assert_array_equal(trainer.act(changed, greedy=True)[camera], original[camera])
        finally:
            trainer.env.close()

    def test_world_model_predicts_residual_and_shared_reward(self):
        model = WorldModel(2, 3, 2, hidden=8, members=2)
        obs, actions = torch.full((4, 2, 3), 10.), torch.zeros(4, 2, 2)
        mean = torch.full((2, 4, 7), 3.)
        mean[..., -1] = 7.
        with patch.object(model, "forward", return_value=(mean, torch.zeros_like(mean), torch.full((2, 4), -100.))), \
             patch("torch.randn_like", side_effect=lambda x: torch.zeros_like(x)):
            next_obs, rewards, terminated = model.predict(obs, actions)
        torch.testing.assert_close(next_obs, obs + 3)
        torch.testing.assert_close(rewards, torch.full((4, 2), 7.))
        self.assertFalse(terminated.any())

    def test_multi_step_rollout_keeps_all_transitions(self):
        trainer = MAMBPOTrainer(small_config(rollout_length=3))
        try:
            starts = torch.zeros(2, 4, trainer.env.obs_dim)
            def prediction(obs, actions):
                return obs + 1, torch.ones(len(obs), 4), torch.zeros(len(obs))
            with patch.object(trainer.model, "predict", side_effect=prediction), \
                 patch.object(trainer.env, "step", side_effect=AssertionError("Model rollout cannot step MATE")):
                self.assertEqual(trainer.rollout_model(starts), 6)
            batch = trainer.model_replay.latest(6)
            torch.testing.assert_close(batch["obs"][:, 0, 0], torch.tensor([0., 0., 1., 1., 2., 2.]))
            torch.testing.assert_close(batch["next_obs"], batch["obs"] + 1)
        finally:
            trainer.env.close()

    def test_rollout_stops_at_predicted_terminal(self):
        trainer = MAMBPOTrainer(small_config(rollout_length=3))
        try:
            starts = torch.zeros(2, 4, trainer.env.obs_dim)
            with patch.object(trainer.model, "predict", return_value=(starts, torch.ones(2, 4), torch.ones(2))):
                self.assertEqual(trainer.rollout_model(starts), 2)
            self.assertEqual(trainer.model_replay.latest(2)["terminated"].sum(), 2)
        finally:
            trainer.env.close()

    def test_critic_bootstrap_uses_own_entropy_and_terminal_mask(self):
        trainer = MAMBPOTrainer(small_config())
        try:
            batch = dict(obs=torch.zeros(2, 4, trainer.env.obs_dim), actions=torch.zeros(2, 4, 2),
                         next_obs=torch.zeros(2, 4, trainer.env.obs_dim), rewards=torch.ones(2, 4),
                         terminated=torch.tensor([1., 0.]))
            with ExitStack() as stack:
                for i, agent in enumerate(trainer.agents):
                    stack.enter_context(patch.object(agent.target_actor, "forward", return_value=(
                        torch.zeros(2, 2), torch.full((2,), -float(i + 1)))))
                    stack.enter_context(patch.object(agent, "q_values", side_effect=lambda o, a, target=False:
                                                     torch.full((2, 2), 10. if target else 0.)))
                stack.enter_context(patch.object(trainer, "_optimize", side_effect=lambda opt, ps, loss, clip: loss.item()))
                actual = trainer.update_critics(batch)
            expected = np.mean([(1 + (1 + trainer.config.gamma * (10 + .1 * (i + 1))) ** 2) / 2 for i in range(4)])
            self.assertAlmostEqual(actual, expected, places=4)
        finally:
            trainer.env.close()

    def test_actor_delay_and_target_updates(self):
        trainer = MAMBPOTrainer(small_config(policy_updates=1, actor_delay=2))
        try:
            trainer.collect_episode(0)
            initial = {k: v.clone() for k, v in trainer.agents[0].actor.state_dict().items()}
            target = {k: v.clone() for k, v in trainer.agents[0].target_actor.state_dict().items()}
            trainer.update()
            self.assertEqual(trainer.actor_updates, 0)
            for k, v in initial.items():
                torch.testing.assert_close(v, trainer.agents[0].actor.state_dict()[k], rtol=0, atol=0)
            trainer.update()
            self.assertEqual(trainer.actor_updates, 1)
            for key, old_target in target.items():
                expected = old_target.lerp(trainer.agents[0].actor.state_dict()[key], trainer.config.tau)
                torch.testing.assert_close(expected, trainer.agents[0].target_actor.state_dict()[key])
        finally:
            trainer.env.close()

    def test_real_mate_learning_and_checkpoint_resume(self):
        trainer = MAMBPOTrainer(small_config(wandb_api="test-secret"))
        try:
            model_initial = [p.detach().clone() for p in trainer.model.parameters()]
            stats = trainer.collect_episode(0, learn=True)
            self.assertTrue(all(np.isfinite(v) for v in stats.values()))
            self.assertEqual(trainer.updates, 4)
            self.assertEqual(trainer.actor_updates, 2)
            self.assertEqual(trainer.model_updates, 1)
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(model_initial, trainer.model.parameters())))
            self.assertEqual(trainer.real_replay.latest(8)["terminated"].sum(), 0)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "latest.pt"
                trainer.save(path)
                self.assertNotIn("wandb_api", torch.load(path, weights_only=True)["config"])
                restored = MAMBPOTrainer.from_checkpoint(path)
                try:
                    self.assertEqual(restored.updates, trainer.updates)
                    self.assertTrue(restored.needs_refit)
                    self.assertEqual(len(restored.real_replay), 0)
                    for name in ("agents", "model"):
                        for key, value in getattr(trainer, name).state_dict().items():
                            torch.testing.assert_close(value, getattr(restored, name).state_dict()[key], rtol=0, atol=0)
                    obs, _ = trainer.env.reset(seed=20)
                    np.testing.assert_array_equal(trainer.act(obs, greedy=True), restored.act(obs, greedy=True))
                    for agent, opts in zip(restored.agents, restored.optimizers):
                        self.assertIs(opts["alpha"].param_groups[0]["params"][0], agent.log_alpha)
                    restored.collect_episode(1, learn=True)
                    self.assertEqual(restored.model_updates, 2)
                    self.assertEqual(restored.updates, 8)
                    self.assertNotEqual(restored.agents[0].log_alpha.item(), trainer.agents[0].log_alpha.item())
                finally:
                    restored.env.close()
        finally:
            trainer.env.close()

    def test_single_camera_fixed_temperature(self):
        trainer = MAMBPOTrainer(small_config(config="MATE-1v2-0.yaml", autotune=False))
        try:
            trainer.collect_episode(0)
            initial = trainer.agents[0].log_alpha.clone()
            self.assertTrue(all(np.isfinite(v) for v in trainer.update().values()))
            torch.testing.assert_close(trainer.agents[0].log_alpha, initial, rtol=0, atol=0)
        finally:
            trainer.env.close()

    def test_invalid_configuration(self):
        for kwargs in ({"actor_delay": 0}, {"real_ratio": 2}, {"ensemble_size": 0},
                       {"model_steps": 0}, {"replay_capacity": 2}, {"alpha": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                small_config(**kwargs)


if __name__ == "__main__":
    unittest.main()
