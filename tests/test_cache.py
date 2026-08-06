import unittest
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SPEC = importlib.util.spec_from_file_location("fbcache_advanced_nodes", Path(__file__).parents[1] / "nodes.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MiniMaxH3FirstBlockCache = MODULE.MiniMaxH3FirstBlockCache
PRESETS = MODULE.PRESETS


class CacheTests(unittest.TestCase):
    def make_cache(self, preset="H3 Fast — 0.10 / max 2"):
        return MiniMaxH3FirstBlockCache(PRESETS[preset], start_sigma=0.9, end_sigma=0.05, block_count=50)

    def full_step(self, cache, sigma=0.8, residual_scale=1.0, shape=(4, 8)):
        x = torch.zeros(shape)
        cache.begin_call(x, torch.tensor([sigma * 1000]), {})
        residual = torch.ones(shape) * residual_scale
        first_output = x + residual
        cache.decide(residual, first_output)
        cache.finish_full_step(first_output + 2)
        cache.end_call()

    def decision(self, cache, sigma=0.7, residual_scale=1.05, shape=(4, 8)):
        x = torch.zeros(shape)
        cache.begin_call(x, torch.tensor([sigma * 1000]), {})
        residual = torch.ones(shape) * residual_scale
        cache.decide(residual, x + residual)
        result = cache.current.use_cache
        if result:
            cache.finish_cached_step(x + residual)
        else:
            cache.finish_full_step(x + residual + 2)
        cache.end_call()
        return result

    def test_three_presets_and_fast_is_middle_threshold(self):
        self.assertEqual(len(PRESETS), 3)
        self.assertEqual(PRESETS["H3 Fast — 0.10 / max 2"].threshold, 0.10)

    def test_cache_hit_and_max_two_consecutive_hits(self):
        cache = self.make_cache()
        self.full_step(cache)
        self.assertTrue(self.decision(cache, sigma=0.7))
        self.assertTrue(self.decision(cache, sigma=0.6))
        self.assertFalse(self.decision(cache, sigma=0.5))

    def test_window_keeps_early_and_late_steps_dense(self):
        cache = self.make_cache()
        self.full_step(cache)
        self.assertFalse(self.decision(cache, sigma=0.95))
        self.assertFalse(self.decision(cache, sigma=0.04))

    def test_shape_change_invalidates_cache(self):
        cache = self.make_cache()
        self.full_step(cache)
        x = torch.zeros((8, 8))
        cache.begin_call(x, torch.tensor([700.0]), {})
        self.assertIsNone(cache.current.previous_first_residual)
        cache.end_call()

    def test_sigma_reversal_invalidates_cache(self):
        cache = self.make_cache()
        self.full_step(cache, sigma=0.7)
        x = torch.zeros((4, 8))
        cache.begin_call(x, torch.tensor([800.0]), {})
        self.assertIsNone(cache.current.previous_first_residual)
        cache.end_call()

    def test_temporal_guard_rejects_local_change_hidden_by_global_mean(self):
        config = MODULE.PresetConfig(0.10, temporal_guard=True)
        cache = MiniMaxH3FirstBlockCache(config, start_sigma=0.9, end_sigma=0.05, block_count=50)
        payload = {"layout": SimpleNamespace(segments=[(0, 4, "video")], signature=(0, 2, 0, 0, 0))}
        x = torch.zeros((4, 2))

        cache.begin_call(x, torch.tensor([800.0]), {}, payload)
        residual = torch.ones_like(x)
        cache.decide(residual, residual)
        cache.finish_full_step(residual + 2)
        cache.end_call()

        cache.begin_call(x, torch.tensor([700.0]), {}, payload)
        changed = residual.clone()
        changed[:2] *= 1.15
        cache.decide(changed, changed)
        self.assertLess(cache.current.last_diff, config.threshold)
        self.assertFalse(cache.current.use_cache)
        cache.end_call()


if __name__ == "__main__":
    unittest.main()
