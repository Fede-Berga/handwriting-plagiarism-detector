import pytest
import numpy as np
from pathlib import Path
from handwriting_detector import FeatureExtractor, HandwritingComparator

class TestFeatureExtractor:
    def test_initialization(self):
        extractor = FeatureExtractor()
        assert len(extractor.feature_names) == 15
    
    def test_binarization(self):
        # Create dummy grayscale image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = FeatureExtractor._binarize(image)
        assert binary.shape == image.shape
        assert set(np.unique(binary)).issubset({0, 255})

class TestComparator:
    def test_similarity_computation(self):
        comparator = HandwritingComparator()
        f1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f2 = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        similarity = comparator._compute_similarity(f1, f2)
        assert 0 <= similarity <= 1
        assert similarity > 0.8  # Should be high for similar vectors

def test_package_import():
    """Test that package can be imported"""
    import handwriting_detector
    assert handwriting_detector.__version__ == "0.1.0"