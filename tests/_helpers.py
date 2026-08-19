import csv
import hashlib
from pathlib import Path

import numpy as np

BASELINE_DIR = Path(__file__).parent / "skeletons"
FEATURE_DIR = Path(__file__).parent / "features"


def skeleton_path(name: str) -> Path:
    return BASELINE_DIR / f"skeleton_{name}.npz"


def feature_path(name: str) -> Path:
    return FEATURE_DIR / f"features_{name}.csv"


def write_feature_csv(path: Path, features: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "value"])
        for key in sorted(features):
            writer.writerow([key, f"{features[key]:.17g}"])


def read_feature_csv(path: Path) -> dict[str, float]:
    loaded: dict[str, float] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["feature", "value"]:
            raise ValueError("invalid feature csv header")
        for row in reader:
            if len(row) != 2:
                raise ValueError("invalid feature csv row")
            key, value = row
            loaded[key] = float(value)
    return loaded


def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()
