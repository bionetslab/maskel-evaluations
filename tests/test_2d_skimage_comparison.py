"""Test comparing maskel.thin with skimage skeletonize(method='lee') on an HRF image."""

import numpy as np
import pytest
from maskel.thin import lee94_thin
from skimage.morphology import skeletonize

from hrf import HRFDataset, preprocess_segmentation

HRF_PATH = "HRF"


@pytest.mark.slow
class TestSkimageComparison2D:
    """Compare maskel.thin with scikit-image skeletonize (Lee) on a single HRF image."""

    @pytest.fixture(scope="class")
    def sample_data(self):
        ds = HRFDataset(HRF_PATH)
        _, seg, mask, _ = ds.load_sample(0)
        cleaned = preprocess_segmentation(seg, mask)
        return cleaned

    def test_maskel_vs_skimage_lee(self, sample_data):
        binary = sample_data > 0
        maskel_skel = lee94_thin(sample_data)
        skimage_skel = skeletonize(binary, method="lee")

        assert maskel_skel.shape == skimage_skel.shape, (
            f"shape mismatch: maskel {maskel_skel.shape} vs skimage {skimage_skel.shape}"
        )
        assert np.array_equal(maskel_skel, skimage_skel), (
            "skeleton mismatch: algorithms produce different results"
        )
