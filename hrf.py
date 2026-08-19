import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image
from scipy import ndimage

from ._utils import to_binary


class HRFDataset:
    """
    Loader for the High-Resolution Fundus (HRF) Image Database.

    45 images total: 15 healthy (h), 15 diabetic retinopathy (dr), 15 glaucoma (g).
    Each has: fundus image (images/), manual segmentation (manual1/), FOV mask (mask/).
    """

    PHENOTYPES: ClassVar[dict[str, str]] = {
        "h": "healthy",
        "dr": "diabetic_retinopathy",
        "g": "glaucoma",
    }

    def __init__(self, hrf_path: str):
        self.hrf_path = Path(hrf_path)
        self.images_path = self.hrf_path / "images"
        self.manual_path = self.hrf_path / "manual1"
        self.mask_path = self.hrf_path / "mask"

        if not self.hrf_path.exists():
            raise ValueError(f"HRF path does not exist: {hrf_path}")

        self.image_list = self._build_image_list()

    def _build_image_list(self) -> list[dict]:
        image_list = []
        for seg_file in sorted(self.manual_path.glob("*.tif")):
            parts = seg_file.stem.split("_")
            if len(parts) != 2:
                continue
            image_id, phenotype_code = parts
            if phenotype_code not in self.PHENOTYPES:
                continue

            img_file = self.images_path / f"{image_id}_{phenotype_code}.JPG"
            if not img_file.exists():
                img_file = self.images_path / f"{image_id}_{phenotype_code}.jpg"
            if not img_file.exists():
                continue

            image_list.append(
                {
                    "id": image_id,
                    "phenotype_code": phenotype_code,
                    "phenotype": self.PHENOTYPES[phenotype_code],
                    "image_path": img_file,
                    "segmentation_path": seg_file,
                    "mask_path": self.mask_path
                    / f"{image_id}_{phenotype_code}_mask.tif",
                    "name": f"{image_id}_{phenotype_code}",
                }
            )
        return image_list

    def load_sample(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        if index < 0 or index >= len(self.image_list):
            raise IndexError(f"Index {index} out of range [0, {len(self.image_list)})")

        info = self.image_list[index]

        image = np.array(Image.open(info["image_path"]))

        seg = np.array(Image.open(info["segmentation_path"]))
        segmentation = to_binary(seg)

        mask = np.array(Image.open(info["mask_path"]))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = to_binary(mask)

        return image, segmentation, mask, info

    def __len__(self) -> int:
        return len(self.image_list)

    def __iter__(self):
        for i in range(len(self)):
            yield self.load_sample(i)

    def summary(self) -> str:
        counts = {p: 0 for p in self.PHENOTYPES.values()}
        for info in self.image_list:
            counts[info["phenotype"]] += 1

        s = f"HRF Dataset Summary\n{'=' * 18}\nTotal images: {len(self.image_list)}\n\nPhenotype distribution:\n"
        for phenotype, count in counts.items():
            s += f"  {phenotype}: {count}\n"
        return s


def preprocess_segmentation(
    segmentation: np.ndarray, mask: np.ndarray | None = None, min_size: int = 50
) -> np.ndarray:
    """Preprocess vessel segmentation for skeletonization."""

    cleaned = segmentation.copy()
    if mask is not None:
        cleaned = cleaned * mask

    if min_size > 0:
        labeled, _ = ndimage.label(cleaned)
        sizes = np.bincount(labeled.ravel())
        small = sizes < min_size
        small[0] = False
        cleaned[small[labeled]] = 0

    return cleaned


if __name__ == "__main__":
    hrf_path = sys.argv[1] if len(sys.argv) > 1 else "HRF"
    print(f"Loading HRF dataset from: {hrf_path}")
    dataset = HRFDataset(hrf_path)
    print(dataset.summary())

    if len(dataset) > 0:
        image, seg, mask, info = dataset.load_sample(0)
        print(f"\nSample: {info['name']}")
        print(f"  Image shape: {image.shape}")
        print(f"  Segmentation shape: {seg.shape}")
        print(f"  Segmentation vessels: {seg.sum()} pixels")
        print(f"  Phenotype: {info['phenotype']}")
