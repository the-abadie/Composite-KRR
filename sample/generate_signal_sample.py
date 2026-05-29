from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(7)
    n_samples = 1000
    output_dir = Path(__file__).resolve().parent

    desc1 = rng.uniform(-2.0, 2.0, size=(n_samples, 3))
    desc2 = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    desc3 = rng.normal(0.0, 1.0, size=(n_samples, 6))

    signal1 = (
        1.4 * np.sin(1.7 * desc1[:, 0])
        + 0.8 * np.cos(2.1 * desc1[:, 1])
        + 0.4 * np.sin(desc1[:, 2] ** 2)
    )
    signal2 = (
        1.3 * _rbf_bump(desc2, center=np.array([0.75, -0.45]), width=0.45)
        - 0.9 * _rbf_bump(desc2, center=np.array([-0.9, 0.85]), width=0.65)
    )
    target_noise = rng.normal(0.0, 0.05, size=n_samples)
    target = signal1 + signal2 + target_noise

    np.save(output_dir / "desc1.npy", desc1)
    np.save(output_dir / "desc2.npy", desc2)
    np.save(output_dir / "desc3.npy", desc3)
    np.save(output_dir / "target1.npy", target)


def _rbf_bump(x: np.ndarray, *, center: np.ndarray, width: float) -> np.ndarray:
    squared_distance = np.sum((x - center) ** 2, axis=1)
    return np.exp(-squared_distance / (2.0 * width**2))


if __name__ == "__main__":
    main()
