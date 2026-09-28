"""Device-selection checks without importing Torch or fitting a model."""
from types import SimpleNamespace
import unittest

from pytorch_backend import resolve_torch_devices


class TorchDeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            device=lambda name: SimpleNamespace(type=name.split(":")[0]),
        )

    def test_auto_device_list_does_not_override_explicit_cuda_requirement(self):
        with self.assertRaisesRegex(RuntimeError, "is_available.*false"):
            resolve_torch_devices(self.torch, "auto", fallback_device="cuda")

    def test_automatic_backend_can_use_cpu(self):
        devices = resolve_torch_devices(self.torch, "auto", fallback_device="auto")
        self.assertEqual([device.type for device in devices], ["cpu"])

    def test_cpu_request_works_without_cuda(self):
        devices = resolve_torch_devices(self.torch, "auto", fallback_device="cpu")
        self.assertEqual([device.type for device in devices], ["cpu"])


if __name__ == "__main__":
    unittest.main()
