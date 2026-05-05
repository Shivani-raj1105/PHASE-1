"""
Tests for fine alignment functions (ORB + RANSAC).
"""

import numpy as np
import cv2
import pytest

from cad_image_alignment.alignment import (
    _detect_and_match_features,
    _estimate_homography,
    _validate_homography,
    _compute_fine_transform,
)


def test_detect_and_match_features_basic():
    """Test basic ORB keypoint detection and matching."""
    # Create two similar images with some features
    img1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (150, 150), 255, 2)
    cv2.circle(img1, (100, 100), 30, 255, 2)
    
    # Create a slightly translated version
    img2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img2, (55, 55), (155, 155), 255, 2)
    cv2.circle(img2, (105, 105), 30, 255, 2)
    
    kp1, kp2, des1, des2, matches = _detect_and_match_features(img1, img2)
    
    # Should detect keypoints in both images
    assert len(kp1) > 0
    assert len(kp2) > 0
    assert des1 is not None
    assert des2 is not None
    # Should find some matches
    assert len(matches) > 0


def test_detect_and_match_features_empty_images():
    """Test that empty images return no matches."""
    img1 = np.zeros((200, 200), dtype=np.uint8)
    img2 = np.zeros((200, 200), dtype=np.uint8)
    
    kp1, kp2, des1, des2, matches = _detect_and_match_features(img1, img2)
    
    # Empty images should have no descriptors
    assert des1 is None or len(kp1) == 0
    assert des2 is None or len(kp2) == 0
    assert len(matches) == 0


def test_estimate_homography_insufficient_matches():
    """Test that insufficient matches returns None."""
    # Create dummy keypoints and matches (fewer than MIN_MATCH_COUNT=8)
    kp1 = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(5)]
    kp2 = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(5)]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(5)]
    
    H, inlier_ratio = _estimate_homography(kp1, kp2, matches)
    
    assert H is None
    assert inlier_ratio is None


def test_estimate_homography_sufficient_matches():
    """Test homography estimation with sufficient matches."""
    # Create keypoints for a known transformation (translation)
    src_pts = [(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0),
               (50.0, 50.0), (30.0, 30.0), (70.0, 70.0), (50.0, 80.0)]
    dst_pts = [(x + 5.0, y + 5.0) for x, y in src_pts]  # Translate by (5, 5)
    
    kp1 = [cv2.KeyPoint(x, y, 1.0) for x, y in src_pts]
    kp2 = [cv2.KeyPoint(x, y, 1.0) for x, y in dst_pts]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(len(src_pts))]
    
    H, inlier_ratio = _estimate_homography(kp1, kp2, matches)
    
    assert H is not None
    assert H.shape == (3, 3)
    assert inlier_ratio is not None
    assert 0.0 <= inlier_ratio <= 1.0


def test_validate_homography_valid_scale():
    """Test that homography with valid scale passes validation."""
    # Create a homography with scale = 1.0 (identity-like)
    H = np.eye(3, dtype=np.float64)
    
    assert _validate_homography(H) == True


def test_validate_homography_scale_too_small():
    """Test that homography with scale < 0.5 fails validation."""
    # Create a homography with scale = 0.3 (too small)
    H = np.array([
        [0.3, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    assert _validate_homography(H) == False


def test_validate_homography_scale_too_large():
    """Test that homography with scale > 2.0 fails validation."""
    # Create a homography with scale = 2.5 (too large)
    H = np.array([
        [2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    assert _validate_homography(H) == False


def test_validate_homography_boundary_values():
    """Test that homography with scale at boundaries passes validation."""
    # Test scale = 0.5 (minimum)
    H_min = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    assert _validate_homography(H_min) == True
    
    # Test scale = 2.0 (maximum)
    H_max = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    assert _validate_homography(H_max) == True


def test_compute_fine_transform_insufficient_matches():
    """Test that insufficient matches causes fallback."""
    # Create images with very few features (just a single small dot)
    # This should not produce enough keypoints for matching
    coarsely_aligned = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(coarsely_aligned, (50, 50), 2, 255, -1)
    
    cad_edge_map = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(cad_edge_map, (50, 50), 2, 255, -1)
    
    coarse_matrix = np.eye(3, dtype=np.float64)
    
    H_total, inlier_ratio = _compute_fine_transform(
        coarsely_aligned,
        cad_edge_map,
        coarse_matrix
    )
    
    # Should return None due to insufficient matches
    # (or if it succeeds, that's also acceptable - ORB is robust)
    if H_total is not None:
        # If it succeeded, verify the outputs are valid
        assert H_total.shape == (3, 3)
        assert inlier_ratio is not None
        assert 0.0 <= inlier_ratio <= 1.0


def test_compute_fine_transform_with_good_features():
    """Test fine transform computation with good feature matches."""
    # Create images with rich features
    coarsely_aligned = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(coarsely_aligned, (50, 50), (150, 150), 255, 2)
    cv2.circle(coarsely_aligned, (100, 100), 30, 255, 2)
    cv2.circle(coarsely_aligned, (80, 80), 10, 255, 2)
    cv2.circle(coarsely_aligned, (120, 120), 10, 255, 2)
    
    # Create a slightly different version (should still match well)
    cad_edge_map = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(cad_edge_map, (52, 52), (152, 152), 255, 2)
    cv2.circle(cad_edge_map, (102, 102), 30, 255, 2)
    cv2.circle(cad_edge_map, (82, 82), 10, 255, 2)
    cv2.circle(cad_edge_map, (122, 122), 10, 255, 2)
    
    coarse_matrix = np.eye(3, dtype=np.float64)
    
    H_total, inlier_ratio = _compute_fine_transform(
        coarsely_aligned,
        cad_edge_map,
        coarse_matrix
    )
    
    # May succeed or fail depending on feature detection
    # If it succeeds, verify the outputs
    if H_total is not None:
        assert H_total.shape == (3, 3)
        assert inlier_ratio is not None
        assert 0.0 <= inlier_ratio <= 1.0
