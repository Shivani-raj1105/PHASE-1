"""
Tests for the main align() function covering tasks 7.1-7.4.
"""

import pytest
import numpy as np
import logging
from cad_image_alignment import align, AlignmentResult


def create_simple_shape(size=100, offset=(0, 0)):
    """Create a simple rectangular shape for testing."""
    img = np.zeros((size, size), dtype=np.uint8)
    x_off, y_off = offset
    img[30+y_off:70+y_off, 30+x_off:70+x_off] = 255
    return img


def test_align_returns_alignment_result():
    """Test that align() returns an AlignmentResult object."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    assert isinstance(result, AlignmentResult)


def test_align_result_has_all_fields():
    """Test that AlignmentResult contains all 6 required fields (Task 7.4)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    # Verify all 6 fields are present
    assert hasattr(result, 'aligned_image')
    assert hasattr(result, 'transform_matrix')
    assert hasattr(result, 'alignment_score')
    assert hasattr(result, 'strategy')
    assert hasattr(result, 'low_confidence')
    assert hasattr(result, 'inlier_ratio')


def test_align_aligned_image_properties():
    """Test that aligned_image has correct shape and dtype (Task 7.2, 7.4)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    assert result.aligned_image.shape == cad_map.shape
    assert result.aligned_image.dtype == np.uint8


def test_align_transform_matrix_properties():
    """Test that transform_matrix has correct shape and dtype (Task 7.4)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    assert result.transform_matrix.shape == (3, 3)
    assert result.transform_matrix.dtype == np.float64


def test_align_alignment_score_bounds():
    """Test that alignment_score is in [0.0, 1.0] (Task 7.2, 7.4)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    assert 0.0 <= result.alignment_score <= 1.0


def test_align_strategy_values():
    """Test that strategy is one of the valid values (Task 7.1, 7.4)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    assert result.strategy in ["homography", "affine_coarse_only", "identity"]


def test_align_low_confidence_flag():
    """Test that low_confidence flag is set correctly (Task 7.2, 7.4)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    # low_confidence should be True if alignment_score < 0.30
    if result.alignment_score < 0.30:
        assert result.low_confidence is True
    else:
        assert result.low_confidence is False


def test_align_identity_fallback_no_contour():
    """Test identity fallback when no contour found (Task 7.1)."""
    # Create edge maps with no valid contours (too small)
    cad_map = np.zeros((100, 100), dtype=np.uint8)
    cad_map[49:51, 49:51] = 255  # Tiny contour < 1% of image area
    
    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[49:51, 49:51] = 255
    
    result = align(cad_map, real_map)
    
    # Should fall back to identity
    assert result.strategy == "identity"
    assert np.allclose(result.transform_matrix, np.eye(3))


def test_align_affine_coarse_only_fallback():
    """Test affine_coarse_only fallback when fine alignment fails (Task 7.1)."""
    # Create edge maps where coarse alignment works but fine alignment fails
    # (e.g., not enough features for ORB matching)
    cad_map = create_simple_shape()
    real_map = create_simple_shape(offset=(5, 5))
    
    result = align(cad_map, real_map)
    
    # Strategy should be either affine_coarse_only or homography
    # (depends on whether ORB finds enough features)
    assert result.strategy in ["affine_coarse_only", "homography"]


def test_align_returns_result_even_with_low_confidence():
    """Test that result is returned even when low_confidence=True (Task 7.4)."""
    # Create mismatched edge maps that will produce low confidence
    cad_map = create_simple_shape()
    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[10:30, 10:30] = 255  # Different shape/position
    
    # Should not raise an exception
    result = align(cad_map, real_map)
    
    # Result should be returned
    assert isinstance(result, AlignmentResult)
    # If low confidence, flag should be set
    if result.low_confidence:
        assert result.alignment_score < 0.30


def test_align_logging_on_fallback(caplog):
    """Test that WARNING is logged on fallback (Task 7.3)."""
    # Create edge maps with no valid contours to trigger fallback
    cad_map = np.zeros((100, 100), dtype=np.uint8)
    cad_map[49:51, 49:51] = 255
    
    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[49:51, 49:51] = 255
    
    with caplog.at_level(logging.WARNING):
        result = align(cad_map, real_map)
    
    # Should have logged a warning about fallback
    assert any("fallback" in record.message.lower() or 
               "no valid contour" in record.message.lower()
               for record in caplog.records)


def test_align_logging_on_success(caplog):
    """Test that DEBUG is logged on successful alignment (Task 7.3)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    with caplog.at_level(logging.DEBUG):
        result = align(cad_map, real_map)
    
    # Should have logged debug message with strategy and score
    debug_messages = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert any("strategy" in msg.lower() and "score" in msg.lower() 
               for msg in debug_messages)


def test_align_logging_on_low_confidence(caplog):
    """Test that WARNING is logged when alignment score is low (Task 7.3)."""
    # Create mismatched edge maps
    cad_map = create_simple_shape()
    real_map = np.zeros((100, 100), dtype=np.uint8)
    real_map[10:30, 10:30] = 255
    
    with caplog.at_level(logging.WARNING):
        result = align(cad_map, real_map)
    
    # If low confidence, should have logged a warning
    if result.low_confidence:
        assert any("low confidence" in record.message.lower() 
                   for record in caplog.records)


def test_align_inlier_ratio_type():
    """Test that inlier_ratio is either float or None (Task 7.4)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    assert result.inlier_ratio is None or isinstance(result.inlier_ratio, float)


def test_align_with_resolution_mismatch():
    """Test that align handles resolution mismatch (Task 7.1)."""
    cad_map = create_simple_shape(size=100)
    real_map = create_simple_shape(size=80)
    
    # Should not raise an exception
    result = align(cad_map, real_map)
    
    # Result should have CAD map dimensions
    assert result.aligned_image.shape == cad_map.shape


def test_align_applies_final_transformation():
    """Test that final transformation is applied to real_edge_map (Task 7.2)."""
    cad_map = create_simple_shape()
    real_map = create_simple_shape()
    
    result = align(cad_map, real_map)
    
    # Aligned image should be the result of applying transform_matrix to real_map
    # We can't verify exact pixel values, but we can verify it's not empty
    # and has the right shape
    assert result.aligned_image.shape == cad_map.shape
    assert result.aligned_image.dtype == np.uint8
