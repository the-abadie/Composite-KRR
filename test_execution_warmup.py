import unittest

from config_schema import ExecutionConfig
from runner import warm_up_execution


class ExecutionWarmupTests(unittest.TestCase):
    def test_cpu_and_disabled_warmup_do_not_import_torch(self):
        from unittest.mock import patch
        with patch("composite_krr.torch_runtime.require_torch", side_effect=AssertionError("unexpected torch use")):
            self.assertEqual(warm_up_execution(ExecutionConfig()), {"seconds": 0., "devices": [], "matrix_size": 0})
            config = ExecutionConfig(backend="pytorch")
            self.assertEqual(warm_up_execution(config)["seconds"], 0.)

    def test_cuda_warmup_preserves_rng(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(17)
        before_cpu = torch.random.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state(0).clone()
        config = ExecutionConfig(backend="pytorch", torch={"device": "cuda:0", "warmup_size": 64})
        result = warm_up_execution(config)
        self.assertGreater(result["seconds"], 0.)
        self.assertEqual(result["devices"], ["cuda:0"])
        self.assertTrue(torch.equal(before_cpu, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state(0)))


if __name__ == "__main__":
    unittest.main()
