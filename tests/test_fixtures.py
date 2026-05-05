"""
Unit tests for synthetic test fixtures.
"""

import numpy as np
import pytest

from tests.fixtures import (
    generate_bracket_shape,
    generate_rotated_variant,
    generate_noisy_variant,
)


def test_generate_bracket_shape_basic():
    """Test that bracket shape generator produces valid edge map."""
    edge_map = generate_bracket_shape()
    
    # Check output type and shape
    assert isinstance(edge_map, np.ndarray)
    assert edge_map.dtype == np.uint8
    assert edge_map.ndim == 2
    assert edge_map.shape == (512, 512)
    
    # Check that edge map contains non-zero pixels
    assert np.any(edge_map > 0), "Edge map should contain edge pixels"
    
    # Check that values are binary (0 or 255)
    unique_values = np.unique(edge_map)
    assert all(v in [0, 255] for v in unique_values), "Edge map should be binary"


def test_generate_bracket_shape_custom_size():
    """Test bracket shape generator with custom size."""
    edge_map = generate_bracket_shape(image_size=(256, 256))
    
    assert edge_map.shape == (256, 256)
    assert edge_map.dtype == np.uint8
    assert np.any(edge_map > 0)


def test_generate_rotated_variant_basic():
    """Test that rotated variant generator produces valid output."""
    # Generate a bracket shape
    edge_map = generate_bracket_shape(image_size=(200, 200))
    
    # Rotate by 45 degrees
    rotated, transform = generate_rotated_variant(edge_map, angle_deg=45.0)
    
    # Check rotated edge map
    assert isinstance(rotated, np.ndarray)
    assert rotated.dtype == np.uint8
    assert rotated.ndim == 2
    assert rotated.shape == edge_map.shape
    assert np.any(rotated > 0), "Rotated edge map should contain edge pixels"
    
    # Check transformation matrix
    assert isinstance(transform, np.ndarray)
    assert transform.dtype == np.float64
    assert transform.shape == (3, 3)
    
    # Check that bottom row is [0, 0, 1] (homogeneous coordinates)
    assert np.allclose(transform[2, :], [0, 0, 1])


def test_generate_rotated_variant_zero_rotation():
    """Test that zero rotation produces identity-like transform."""
    edge_map = generate_bracket_shape(image_size=(200, 200))
    
    rotated, transform = generate_rotated_variant(edge_map, angle_deg=0.0)
    
    # Rotated image should be very similar to original
    # (may have minor differences due to interpolation)
    assert rotated.shape == edge_map.shape
    
    # Transform should be close to identity
    # The 2×3 rotation matrix for 0° should be [[1, 0, 0], [0, 1, 0]]
    # In 3×3 form: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    expected = np.eye(3, dtype=np.float64)
    # Allow for small numerical errors
    assert np.allclose(transform, expected, atol=1e-6)


def test_generate_rotated_variant_multiple_angles():
    """Test rotation at multiple standard angles."""
    edge_map = generate_bracket_shape(image_size=(200, 200))
    
    for angle in [0, 45, 90, 135, 180, 270]:
        rotated, transform = generate_rotated_variant(edge_map, angle_deg=float(angle))
        
        assert rotated.shape == edge_map.shape
        assert rotated.dtype == np.uint8
        assert transform.shape == (3, 3)
        assert transform.dtype == np.float64


def test_generate_noisy_variant_basic():
    """Test that noisy variant generator produces valid output."""
    edge_map = generate_bracket_shape(image_size=(200, 200))
    
    # Generate noisy variant with 2% noise
    noisy = generate_noisy_variant(edge_map, noise_level=0.02)
    
    # Check output type and shape
    assert isinstance(noisy, np.ndarray)
    assert noisy.dtype == np.uint8
    assert noisy.ndim == 2
    assert noisy.shape == edge_map.shape
    
    # Check that noisy image is different from original
    assert not np.array_equal(noisy, edge_map), "Noisy image should differ from original"


def test_generate_noisy_variant_zero_noise():
    """Test that zero noise level produces identical output."""
    edge_map = generate_bracket_shape(image_size=(200, 200))
    
    # Generate with zero noise
    noisy = generate_noisy_variant(edge_map, noise_level=0.0)
    
    # Should be identical to original
    assert np.array_equal(noisy, edge_map)


def test_generate_noisy_variant_high_noise():
    """Test that high noise level produces significant changes."""
    edge_map = generate_bracket_shape(image_size=(200, 200))
    
    # Generate with 10% noise
    noisy = generate_noisy_variant(edge_map, noise_level=0.10)
    
    # Count number of different pixels
    diff_pixels = np.sum(noisy != edge_map)
    total_pixels = edge_map.shape[0] * edge_map.shape[1]
    
    # Should have some different pixels (noise may overlap with existing values)
    # We expect at least 2% actual difference (since some noise overlaps with existing values)
    assert diff_pixels > 0.02 * total_pixels, \
        f"Expected at least {0.02 * total_pixels:.0f} different pixels, got {diff_pixels}"
    
    # But not too many (should be less than the full 10% since some overlaps)
    assert diff_pixels < 0.15 * total_pixels, \
        f"Expected less than {0.15 * total_pixels:.0f} different pixels, got {diff_pixels}"


def test_generate_noisy_variant_preserves_binary():
    """Test that noisy variant maintains binary values."""
    edge_map = generate_bracket_shape(image_size=(200, 200))
    
    noisy = generate_noisy_variant(edge_map, noise_level=0.05)
    
    # Check that values are still binary (0 or 255)
    unique_values = np.unique(noisy)
    assert all(v in [0, 255] for v in unique_values), "Noisy edge map should remain binary"


def test_fixtures_integration():
    """Integration test: generate bracket, rotate, and add noise."""
    # Generate bracket shape
    bracket = generate_bracket_shape(image_size=(300, 300))
    assert np.any(bracket > 0)
    
    # Rotate by 90 degrees
    rotated, transform = generate_rotated_variant(bracket, angle_deg=90.0)
    assert rotated.shape == bracket.shape
    assert transform.shape == (3, 3)
    
    # Add noise
    noisy = generate_noisy_variant(rotated, noise_level=0.03)
    assert noisy.shape == rotated.shape
    assert not np.array_equal(noisy, rotated)
    
    # All outputs should be valid edge maps
    for img in [bracket, rotated, noisy]:
        assert img.dtype == np.uint8
        assert img.ndim == 2
        assert np.any(img > 0)
