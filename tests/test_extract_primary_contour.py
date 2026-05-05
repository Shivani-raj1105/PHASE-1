"""
Unit tests for _extract_primary_contour function.
"""

import numpy as np
import cv2
import pytest
from cad_image_alignment.alignment import _extract_primary_contour


def test_extract_primary_contour_simple_shape():
    """Test extraction of primary contour from a simple rectangular shape."""
    # Create a 200x200 image with a 100x100 white rectangle in the center
    edge_map = np.zeros((200, 200), dtype=np.uint8)
    edge_map[50:150, 50:150] = 255
    
    # Extract edges to create contours
    edge_map = cv2.Canny(edge_map, 50, 150)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None, "Should find a contour in the edge map"
    assert result.contour is not None
    assert len(result.centroid) == 2
    assert len(result.bbox) == 4
    assert result.bbox_diagonal > 0


def test_extract_primary_contour_multiple_contours():
    """Test that the largest contour is selected when multiple contours exist."""
    # Create a 300x300 image with two rectangles of different sizes
    edge_map = np.zeros((300, 300), dtype=np.uint8)
    
    # Small rectangle (20x20)
    cv2.rectangle(edge_map, (10, 10), (30, 30), 255, 1)
    
    # Large rectangle (100x100)
    cv2.rectangle(edge_map, (100, 100), (200, 200), 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None, "Should find a contour"
    # The larger contour should be selected
    area = cv2.contourArea(result.contour)
    assert area > 1000, "Should select the larger contour"


def test_extract_primary_contour_no_valid_contour():
    """Test that None is returned when no contour meets the minimum area threshold."""
    # Create a 1000x1000 image with a tiny 5x5 rectangle (0.0025% of image area)
    edge_map = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.rectangle(edge_map, (10, 10), (15, 15), 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    # Should return None because the contour is < 1% of image area
    assert result is None, "Should return None for contours below 1% threshold"


def test_extract_primary_contour_empty_image():
    """Test that None is returned for an empty edge map."""
    edge_map = np.zeros((200, 200), dtype=np.uint8)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is None, "Should return None for empty edge map"


def test_extract_primary_contour_centroid_calculation():
    """Test that centroid is calculated correctly."""
    # Create a 200x200 image with a rectangle
    edge_map = np.zeros((200, 200), dtype=np.uint8)
    # Rectangle from (50, 50) to (150, 150) - center should be around (100, 100)
    cv2.rectangle(edge_map, (50, 50), (150, 150), 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None
    cx, cy = result.centroid
    # Centroid should be approximately at the center of the rectangle
    assert 90 < cx < 110, f"Centroid x={cx} should be near 100"
    assert 90 < cy < 110, f"Centroid y={cy} should be near 100"


def test_extract_primary_contour_bbox_calculation():
    """Test that bounding box is calculated correctly."""
    # Create a 200x200 image with a rectangle
    edge_map = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(edge_map, (50, 60), (150, 140), 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None
    x, y, w, h = result.bbox
    
    # Bounding box should approximately match the rectangle
    assert 45 < x < 55, f"Bbox x={x} should be near 50"
    assert 55 < y < 65, f"Bbox y={y} should be near 60"
    assert 95 < w < 105, f"Bbox width={w} should be near 100"
    assert 75 < h < 85, f"Bbox height={h} should be near 80"


def test_extract_primary_contour_diagonal_calculation():
    """Test that bbox diagonal is calculated correctly."""
    # Create a 200x200 image with a rectangle
    edge_map = np.zeros((200, 200), dtype=np.uint8)
    # Rectangle with width=60, height=80
    cv2.rectangle(edge_map, (50, 50), (110, 130), 255, 1)
    
    result = _extract_primary_contour(edge_map)
    
    assert result is not None
    # Diagonal should be sqrt(60^2 + 80^2) = sqrt(3600 + 6400) = sqrt(10000) = 100
    expected_diagonal = np.sqrt(60**2 + 80**2)
    assert abs(result.bbox_diagonal - expected_diagonal) < 5, \
        f"Diagonal {result.bbox_diagonal} should be near {expected_diagonal}"
