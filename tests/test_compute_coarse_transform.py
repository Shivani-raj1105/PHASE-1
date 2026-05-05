"""
Tests for _compute_coarse_transform() orchestration function.
"""

import numpy as np
import pytest
import cv2

from cad_image_alignment.alignment import _compute_coarse_transform


def create_simple_shape(size=200, offset=(0, 0)):
    """Create a simple rectangular shape for testing."""
    img = np.zeros((size, size), dtype=np.uint8)
    x_off, y_off = offset
    cv2.rectangle(img, (50 + x_off, 50 + y_off), (150 + x_off, 150 + y_off), 255, 2)
    return img


def test_compute_coarse_transform_basic():
    """Test basic coarse transform computation with valid contours."""
    # Create CAD edge map
    cad_edge_map = create_simple_shape(size=200)
    
    # Create real edge map (same shape, should produce identity-like transform)
    real_edge_map = create_simple_shape(size=200)
    
    # Compute coarse transform
    result = _compute_coarse_transform(cad_edge_map, real_edge_map)
    
    # Should return a 3x3 matrix
    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64
    
    # Bottom row should be [0, 0, 1] for affine transform
    np.testing.assert_array_almost_equal(result[2, :], [0.0, 0.0, 1.0])


def test_compute_coarse_transform_with_translation():
    """Test coarse transform with translated shape."""
    # Create CAD edge map
    cad_edge_map = create_simple_shape(size=200, offset=(0, 0))
    
    # Create real edge map with translation
    real_edge_map = create_simple_shape(size=200, offset=(20, 30))
    
    # Compute coarse transform
    result = _compute_coarse_transform(cad_edge_map, real_edge_map)
    
    # Should return a 3x3 matrix
    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64


def test_compute_coarse_transform_no_contour_in_real():
    """Test that None is returned when no contour found in real edge map."""
    # Create CAD edge map with valid contour
    cad_edge_map = create_simple_shape(size=200)
    
    # Create empty real edge map (no contours)
    real_edge_map = np.zeros((200, 200), dtype=np.uint8)
    
    # Compute coarse transform
    result = _compute_coarse_transform(cad_edge_map, real_edge_map)
    
    # Should return None
    assert result is None


def test_compute_coarse_transform_no_contour_in_cad():
    """Test that None is returned when no contour found in CAD edge map."""
    # Create empty CAD edge map (no contours)
    cad_edge_map = np.zeros((200, 200), dtype=np.uint8)
    
    # Create real edge map with valid contour
    real_edge_map = create_simple_shape(size=200)
    
    # Compute coarse transform
    result = _compute_coarse_transform(cad_edge_map, real_edge_map)
    
    # Should return None
    assert result is None


def test_compute_coarse_transform_with_scale():
    """Test coarse transform with scaled shape."""
    # Create CAD edge map with larger rectangle
    cad_edge_map = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(cad_edge_map, (50, 50), (250, 250), 255, 2)
    
    # Create real edge map with smaller rectangle
    real_edge_map = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(real_edge_map, (100, 100), (200, 200), 255, 2)
    
    # Compute coarse transform
    result = _compute_coarse_transform(cad_edge_map, real_edge_map)
    
    # Should return a 3x3 matrix
    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64
    
    # Bottom row should be [0, 0, 1] for affine transform
    np.testing.assert_array_almost_equal(result[2, :], [0.0, 0.0, 1.0])


def test_compute_coarse_transform_with_rotation():
    """Test coarse transform with rotated shape."""
    # Create CAD edge map
    cad_edge_map = create_simple_shape(size=200)
    
    # Create real edge map with rotation
    real_edge_map = np.zeros((200, 200), dtype=np.uint8)
    center = (100, 100)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    temp = create_simple_shape(size=200)
    real_edge_map = cv2.warpAffine(temp, M, (200, 200))
    
    # Compute coarse transform
    result = _compute_coarse_transform(cad_edge_map, real_edge_map)
    
    # Should return a 3x3 matrix
    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64
    
    # Bottom row should be [0, 0, 1] for affine transform
    np.testing.assert_array_almost_equal(result[2, :], [0.0, 0.0, 1.0])
