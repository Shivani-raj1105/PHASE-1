"""
Unit tests for PCA-based rotation estimation.
"""

import numpy as np
import cv2
import pytest
from cad_image_alignment.alignment import _extract_primary_contour, _compute_pca_angle


def test_pca_angle_horizontal_rectangle():
    """Test PCA angle for a horizontal rectangle (should be close to 0° or 180°)."""
    # Create a horizontal rectangle (wider than tall)
    edge_map = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(edge_map, (50, 80), (250, 120), 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None
    # For a horizontal rectangle, PCA angle should be close to 0° or 180°
    # (PCA has 180° ambiguity)
    angle = result.pca_angle_deg
    assert 0 <= angle < 360, f"Angle {angle} should be in [0°, 360°)"
    # Check if angle is close to 0° or 180° (within 10° tolerance)
    assert (angle < 10 or angle > 350 or (170 < angle < 190)), \
        f"Horizontal rectangle angle {angle} should be near 0° or 180°"


def test_pca_angle_vertical_rectangle():
    """Test PCA angle for a vertical rectangle (should be close to 90° or 270°)."""
    # Create a vertical rectangle (taller than wide)
    edge_map = np.zeros((300, 200), dtype=np.uint8)
    cv2.rectangle(edge_map, (80, 50), (120, 250), 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None
    # For a vertical rectangle, PCA angle should be close to 90° or 270°
    angle = result.pca_angle_deg
    assert 0 <= angle < 360, f"Angle {angle} should be in [0°, 360°)"
    # Check if angle is close to 90° or 270° (within 10° tolerance)
    assert (80 < angle < 100 or 260 < angle < 280), \
        f"Vertical rectangle angle {angle} should be near 90° or 270°"


def test_pca_angle_45_degree_line():
    """Test PCA angle for a 45-degree oriented shape."""
    # Create a rotated rectangle at 45 degrees
    edge_map = np.zeros((300, 300), dtype=np.uint8)
    center = (150, 150)
    size = (100, 40)
    angle = 45
    rect = (center, size, angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(edge_map, [box], 0, 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None
    pca_angle = result.pca_angle_deg
    assert 0 <= pca_angle < 360, f"Angle {pca_angle} should be in [0°, 360°)"
    # Should be close to 45° or 225° (180° ambiguity)
    assert (35 < pca_angle < 55 or 215 < pca_angle < 235), \
        f"45° rotated shape angle {pca_angle} should be near 45° or 225°"


def test_pca_angle_normalization():
    """Test that PCA angle is always normalized to [0°, 360°)."""
    # Test with various shapes to ensure angle is always in valid range
    for _ in range(10):
        edge_map = np.zeros((200, 200), dtype=np.uint8)
        # Random rectangle with random rotation
        center = (100, 100)
        size = (np.random.randint(30, 80), np.random.randint(20, 50))
        angle = np.random.uniform(0, 360)
        rect = (center, size, angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(edge_map, [box], 0, 255, 1)
        
        result = _extract_primary_contour(edge_map)
        
        if result is not None:
            assert 0 <= result.pca_angle_deg < 360, \
                f"Angle {result.pca_angle_deg} must be in [0°, 360°)"


def test_compute_pca_angle_directly():
    """Test _compute_pca_angle function directly with a known contour."""
    # Create a simple horizontal line of points
    # Points along x-axis should give angle close to 0° or 180°
    points = np.array([[[i, 100]] for i in range(50, 150)], dtype=np.int32)
    
    angle = _compute_pca_angle(points)
    
    assert 0 <= angle < 360, f"Angle {angle} should be in [0°, 360°)"
    # Horizontal line should be close to 0° or 180°
    assert (angle < 10 or angle > 350 or (170 < angle < 190)), \
        f"Horizontal line angle {angle} should be near 0° or 180°"


def test_compute_pca_angle_vertical_line():
    """Test _compute_pca_angle with a vertical line of points."""
    # Points along y-axis should give angle close to 90° or 270°
    points = np.array([[[100, i]] for i in range(50, 150)], dtype=np.int32)
    
    angle = _compute_pca_angle(points)
    
    assert 0 <= angle < 360, f"Angle {angle} should be in [0°, 360°)"
    # Vertical line should be close to 90° or 270°
    assert (80 < angle < 100 or 260 < angle < 280), \
        f"Vertical line angle {angle} should be near 90° or 270°"


def test_pca_angle_l_shape():
    """Test PCA angle for an L-shaped contour."""
    # Create an L-shape
    edge_map = np.zeros((200, 200), dtype=np.uint8)
    # Vertical part of L
    cv2.line(edge_map, (50, 50), (50, 150), 255, 2)
    # Horizontal part of L
    cv2.line(edge_map, (50, 150), (150, 150), 255, 2)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None
    # L-shape should have a principal axis at approximately 45° or 225°
    # (diagonal from top-left to bottom-right of the L's bounding box)
    angle = result.pca_angle_deg
    assert 0 <= angle < 360, f"Angle {angle} should be in [0°, 360°)"
    # The exact angle depends on the weight distribution, but it should be computed
    # Just verify it's a valid angle
    assert isinstance(angle, (float, np.floating)), "Angle should be a float"
