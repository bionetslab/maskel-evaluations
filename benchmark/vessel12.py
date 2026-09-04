import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import itk
import numpy as np

VESSEL12_MASK_URL = (
    "https://zenodo.org/records/8055066/files/"
    "VESSEL12_01-20_Lungmasks.tar.bz2?download=1"
)


def ensure_vessel12(vessel12_path: Path) -> None:
    """Download the VESSEL12 lung masks from Zenodo into vessel12_path, unless
    already present."""
    if vessel12_path.is_dir() and any(vessel12_path.glob("VESSEL12_*.mhd")):
        return

    vessel12_path.mkdir(parents=True, exist_ok=True)
    print("Downloading VESSEL12 lung masks from Zenodo...")
    tgz_path = vessel12_path / "VESSEL12_01-20_Lungmasks.tar.bz2"
    urlretrieve(VESSEL12_MASK_URL, tgz_path)
    print("Extracting...")
    with tarfile.open(tgz_path, "r:bz2") as tar:
        tar.extractall(path=vessel12_path)
    tgz_path.unlink()
    print("Done.")


class Vessel12Dataset:
    """
    Loader for the VESSEL12 lung CT lung-mask scans.

    20 scans total (VESSEL12_01 .. VESSEL12_20), each a binarized lung mask volume.
    """

    def __init__(self, vessel12_path: str):
        self.vessel12_path = Path(vessel12_path)

        if not self.vessel12_path.exists():
            raise ValueError(f"Vessel12 path does not exist: {vessel12_path}")

        self.scan_list = self._build_scan_list()

    def _build_scan_list(self) -> list[dict]:
        scan_list = []
        for mhd_path in sorted(self.vessel12_path.glob("VESSEL12_*.mhd")):
            scan_list.append({"name": mhd_path.stem, "scan_path": mhd_path})
        return scan_list

    def load_sample(self, index: int) -> tuple[np.ndarray, dict]:
        if index < 0 or index >= len(self.scan_list):
            raise IndexError(f"Index {index} out of range [0, {len(self.scan_list)})")

        info = self.scan_list[index]
        vol = np.asarray(itk.imread(str(info["scan_path"])))
        binarized = (vol > 0).astype(np.uint8)
        return binarized, info

    def __len__(self) -> int:
        return len(self.scan_list)

    def __iter__(self):
        for i in range(len(self)):
            yield self.load_sample(i)

    def summary(self) -> str:
        return (
            f"Vessel12 Dataset Summary\n{'=' * 24}\nTotal scans: {len(self.scan_list)}\n"
        )


if __name__ == "__main__":
    vessel12_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/Vessel12")
    ensure_vessel12(vessel12_path)
    print(f"Loading Vessel12 dataset from: {vessel12_path}")
    dataset = Vessel12Dataset(vessel12_path)
    print(dataset.summary())

    if len(dataset) > 0:
        vol, info = dataset.load_sample(0)
        print(f"\nSample: {info['name']}")
        print(f"  Volume shape: {vol.shape}")
        print(f"  Foreground voxels: {vol.sum()}")
