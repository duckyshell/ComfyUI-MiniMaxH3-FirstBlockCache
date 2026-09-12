import unittest
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location("fbcache_advanced_nodes", Path(__file__).parents[1] / "nodes.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MiniMaxH3FirstBlockCache = MODULE.MiniMaxH3FirstBlockCache
PRESETS = MODULE.PRESETS


class CacheTests(unittest.TestCase):
    def test_persistent_storage_is_allocated_while_graph_paused(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        paused = False
        allocations = {}

        @contextmanager
        def pause():
            nonlocal paused
            paused = True
            try:
                yield
            finally:
                paused = False

        class TrackAllocations(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if isinstance(out, torch.Tensor):
                    allocations.setdefault(out.untyped_storage().data_ptr(), paused)
                return out

        cache = self.make_cache()
        x = torch.zeros(4, 8)
        residual = torch.ones_like(x)
        output = x + residual
        cache.begin_call(x, torch.tensor([800.0]), {})
        with patch.object(MODULE, "_pause_malloc_graph", pause), TrackAllocations():
            cache.decide(residual, output)
            context = cache.current
            for tensor in (context.first_block_output, context.pending_first_residual):
                self.assertTrue(allocations[tensor.untyped_storage().data_ptr()])
            self.assertNotEqual(context.pending_first_residual.data_ptr(), residual.data_ptr())
            cache.finish_full_step(output + 2)
            for tensor in (context.remaining_blocks_residual, context.previous_first_residual):
                self.assertTrue(allocations[tensor.untyped_storage().data_ptr()])
            expected = output + 2
            cached_output = cache.finish_cached_step(output)
            self.assertIs(cached_output, output)
        torch.testing.assert_close(context.previous_first_residual, residual)
        torch.testing.assert_close(cached_output, expected)
        cache.end_call()
        cache.reset()
        self.assertFalse(cache.contexts)
        self.assertIsNone(context.previous_first_residual)
        self.assertIsNone(context.remaining_blocks_residual)

    def test_older_comfy_without_pause_api_or_module(self):
        for prefetch in (None, SimpleNamespace()):
            with self.subTest(prefetch=prefetch), patch.dict(sys.modules, {"comfy.model_prefetch": prefetch}):
                cache = self.make_cache()
                self.full_step(cache)
                self.assertTrue(self.decision(cache))
                cache.reset()

    def test_sample_exception_releases_pending_cache(self):
        cache = self.make_cache()
        context = None

        def fail():
            nonlocal context
            cache.begin_call(torch.zeros(4, 8), torch.tensor([800.0]), {})
            cache.decide(torch.ones(4, 8), torch.ones(4, 8))
            context = cache.current
            raise RuntimeError("interrupted block")

        with self.assertRaisesRegex(RuntimeError, "interrupted block"):
            MODULE.make_sample_wrapper(cache, "test")(fail)
        self.assertFalse(cache.contexts)
        self.assertIsNone(cache.current)
        self.assertIsNone(context.first_block_output)
        self.assertIsNone(context.pending_first_residual)

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
