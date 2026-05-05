"""
Unit tests for _build_affine_matrix function.
"""

import numpy as np
import pytest

from cad_image_alignment.alignment import _build_affine_matrix


def test_build_affine_matrix_identity():
    """Test that scale=1, angle=0, same centroids produces identity-like matrix."""
    scale = 1.0
    angle_deg = 0.0
    real_centroid = (100.0, 100.0)
    cad_centroid = (100.0, 100.0)
    
    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)
    
    # Check shape and dtype
    assert M.shape == (3, 3)
    assert M.dtype == np.float64
    
    # Check bottom row is [0, 0, 1]
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])
    
    # Should be close to identity
    np.testing.assert_array_almost_equal(M, np.eye(3))


def test_build_affine_matrix_scale_only():
    """Test scaling about real centroid."""
    scale = 2.0
    angle_deg = 0.0
    real_centroid = (50.0, 50.0)
    cad_centroid = (50.0, 50.0)
    
    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)
    
    # Check bottom row
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])
    
    # Apply to a point offset from centroid
    point = np.array([60.0, 60.0, 1.0])  # 10 pixels away from centroid
    transformed = M @ point
    
    # After scaling by 2 about (50, 50), point (60, 60) should move to (70, 70)
    # Distance from centroid doubles: 10 -> 20
    expected = np.array([70.0, 70.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected)


def test_build_affine_matrix_rotation_only():
    """Test rotation about real centroid."""
    scale = 1.0
    angle_deg = 90.0  # 90 degrees counterclockwise
    real_centroid = (100.0, 100.0)
    cad_centroid = (100.0, 100.0)
    
    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)
    
    # Check bottom row
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])
    
    # Apply to a point to the right of centroid
    point = np.array([110.0, 100.0, 1.0])  # 10 pixels to the right
    transformed = M @ point
    
    # After 90° rotation, should be 10 pixels above centroid
    expected = np.array([100.0, 110.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)


def test_build_affine_matrix_translation_only():
    """Test translation from real centroid to CAD centroid."""
    scale = 1.0
    angle_deg = 0.0
    real_centroid = (50.0, 50.0)
    cad_centroid = (150.0, 200.0)
    
    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)
    
    # Check bottom row
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])
    
    # Apply to real centroid
    point = np.array([50.0, 50.0, 1.0])
    transformed = M @ point
    
    # Should map to CAD centroid
    expected = np.array([150.0, 200.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected)


def test_build_affine_matrix_combined():
    """Test combined scale, rotation, and translation."""
    scale = 2.0
    angle_deg = 45.0
    real_centroid = (100.0, 100.0)
    cad_centroid = (200.0, 200.0)
    
    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)
    
    # Check shape, dtype, and bottom row
    assert M.shape == (3, 3)
    assert M.dtype == np.float64
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])
    
    # Apply to real centroid - should map to CAD centroid
    point = np.array([100.0, 100.0, 1.0])
    transformed = M @ point
    expected = np.array([200.0, 200.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)


def test_build_affine_matrix_180_degree_rotation():
    """Test 180 degree rotation (for ambiguity resolution)."""
    scale = 1.0
    angle_deg = 180.0
    real_centroid = (100.0, 100.0)
    cad_centroid = (100.0, 100.0)
    
    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)
    
    # Check bottom row
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])
    
    # Apply to a point
    point = np.array([110.0, 110.0, 1.0])  # 10 pixels away diagonally
    transformed = M @ point
    
    # After 180° rotation about (100, 100), should be at (90, 90)
    expected = np.array([90.0, 90.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)


def test_build_affine_matrix_arbitrary_angle():
    """Test with arbitrary angle in [0, 360) range."""
    scale = 1.5
    angle_deg = 237.5
    real_centroid = (75.0, 125.0)
    cad_centroid = (300.0, 400.0)
    
    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)
    
    # Check basic properties
    assert M.shape == (3, 3)
    assert M.dtype == np.float64
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])
    
    # Centroid should map correctly
    point = np.array([75.0, 125.0, 1.0])
    transformed = M @ point
    expected = np.array([300.0, 400.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)
