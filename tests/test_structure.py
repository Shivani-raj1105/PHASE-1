"""
Basic structure tests to verify the module is set up correctly.
"""

import pytest
import numpy as np
from cad_image_alignment import align, apply_transform, AlignmentResult


def test_module_imports():
    """Test that the module and its public API can be imported."""
    assert align is not None
    assert apply_transform is not None
    assert AlignmentResult is not None


def test_alignment_result_dataclass():
    """Test that AlignmentResult can be instantiated."""
    result = AlignmentResult(
        aligned_image=np.zeros((100, 100), dtype=np.uint8),
        transform_matrix=np.eye(3, dtype=np.float64),
        alignment_score=0.85,
        strategy="homography",
        low_confidence=False,
        inlier_ratio=0.75,
    )
    
    assert result.aligned_image.shape == (100, 100)
    assert result.transform_matrix.shape == (3, 3)
    assert result.alignment_score == 0.85
    assert result.strategy == "homography"
    assert result.low_confidence is False
    assert result.inlier_ratio == 0.75


def test_align_basic_functionality():
    """Test that align() returns a valid AlignmentResult."""
    # Create simple edge maps with some content
    cad_map = np.zeros((100, 100), dtype=np.uint8)
    cad_map[40:60, 40:60] = 255
    
    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[40:60, 40:60] = 255
    
    # Call align() - should not raise an exception
    result = align(cad_map, real_map)
    
    # Verify result structure
    assert isinstance(result, AlignmentResult)
    assert result.aligned_image.shape == cad_map.shape
    assert result.aligned_image.dtype == np.uint8
    assert result.transform_matrix.shape == (3, 3)
    assert result.transform_matrix.dtype == np.float64
    assert 0.0 <= result.alignment_score <= 1.0
    assert result.strategy in ["homography", "affine_coarse_only", "identity"]
    assert isinstance(result.low_confidence, bool)
    assert result.inlier_ratio is None or isinstance(result.inlier_ratio, float)


def test_apply_transform_implemented():
    """Test that apply_transform() works correctly."""
    edge_map = np.zeros((100, 100), dtype=np.uint8)
    edge_map[40:60, 40:60] = 255
    matrix = np.eye(3, dtype=np.float64)
    
    # Should not raise an exception
    result = apply_transform(edge_map, matrix)
    
    # Verify result has correct shape and dtype
    assert result.shape == edge_map.shape
    assert result.dtype == np.uint8
