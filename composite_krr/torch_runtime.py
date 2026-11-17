"""Lazy PyTorch loading, dtype and device selection for fitting and scoring."""
from importlib import import_module
from typing import Any

import numpy as np


def require_torch():
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the PyTorch KRR backend. Install a PyTorch "
            "build appropriate for your hardware: CUDA builds for NVIDIA GPUs, "
            "or ROCm builds for supported AMD GPUs. PyTorch exposes both as "
            '`device="cuda"` when available.'
        ) from exc


def resolve_torch_device(torch, device: str | None = "auto"):
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            'Requested PyTorch device "cuda", but torch.cuda.is_available() is false. '
            "For AMD GPUs, install a ROCm-enabled PyTorch build; it still uses "
            'device type "cuda".'
        )
    return resolved


def resolve_torch_devices(
    torch,
    devices=None,
    *,
    fallback_device: str | None = "auto",
    max_devices: int | None = None,
) -> list[Any]:
    if devices is None:
        resolved_devices = [resolve_torch_device(torch, fallback_device)]
    elif isinstance(devices, str):
        if devices == "auto":
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                resolved_devices = [
                    torch.device(f"cuda:{index}")
                    for index in range(device_count)
                ]
            else:
                resolved_devices = [resolve_torch_device(torch, fallback_device)]
        else:
            raw_devices = [device.strip() for device in devices.split(",")]
            raw_devices = [device for device in raw_devices if device]
            if not raw_devices:
                raise ValueError("PyTorch device list cannot be empty.")
            resolved_devices = [
                resolve_torch_device(torch, device)
                for device in raw_devices
            ]
    else:
        raw_devices = list(devices)
        if not raw_devices:
            raise ValueError("PyTorch device list cannot be empty.")
        resolved_devices = [
            resolve_torch_device(torch, device)
            for device in raw_devices
        ]

    if max_devices is not None:
        resolved_devices = resolved_devices[:max_devices]
    if not resolved_devices:
        raise ValueError("No PyTorch devices resolved for cached scoring.")

    return resolved_devices


def resolve_torch_dtype(torch, dtype=None):
    if dtype is None:
        return torch.float64

    if dtype in {torch.float64, torch.float32}:
        return dtype

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return torch.float64
    if dtype == np.dtype(np.float32):
        return torch.float32
    raise ValueError(f"Unsupported PyTorch floating dtype: {dtype}.")


def as_torch_tensor(x, *, torch, device, dtype):
    if hasattr(torch, "is_tensor") and torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)

    return torch.as_tensor(x, dtype=dtype, device=device)


def resolve_torch(*, dtype, device):
    torch = require_torch()
    return torch, resolve_torch_dtype(torch, dtype), resolve_torch_device(torch, device)
