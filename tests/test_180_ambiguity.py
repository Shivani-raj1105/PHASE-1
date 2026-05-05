"""
Tests for 180° ambiguity resolution in coarse alignment.
"""

import numpy as np
import cv2
import pytest

from cad_image_alignment.alignment import (
    _resolve_180_ambiguity,
    apply_transform,
    _compute_alignment_score,
)


def create_simple_shape(size=200):
    """
    Create a simple asymmetric shape (L-shape) for testing.
    This shape has a clear orientation that can be distinguished at 180°.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Draw an L-shape (vertical bar + horizontal bar)
    # Vertical bar on the left
    cv2.rectangle(img, (50, 50), (80, 150), 255, -1)
    # Horizontal bar at the bottom
    cv2.rectangle(img, (50, 120), (150, 150), 255, -1)
    
    # Apply edge detection to get edge map
    edges = cv2.Canny(img, 50, 150)
    
    return edges


def test_resolve_180_ambiguity_selects_correct_orientation():
    """
    Test that _resolve_180_ambiguity correctly selects the orientation
    that produces the higher alignment score.
    """
    # Create a CAD edge map with an L-shape
    cad_edge_map = create_simple_shape(size=200)
    
    # Create a real edge map that is the same as CAD (no rotation)
    real_edge_map = cad_edge_map.copy()
    
    # Define centroids (center of image)
    cad_centroid = (100.0, 100.0)
    real_centroid = (100.0, 100.0)
    
    # No scale difference
    scale = 1.0
    
    # Test with a PCA angle of 0° (correct orientation)
    pca_angle_deg = 0.0
    
    # Call _resolve_180_ambiguity
    best_M = _resolve_180_ambiguity(
        real_edge_map=real_edge_map,
        cad_edge_map=cad_edge_map,
        pca_angle_deg=pca_angle_deg,
        scale=scale,
        real_centroid=real_centroid,
        cad_centroid=cad_centroid,
    )
    
    # Apply the best transformation
    aligned = apply_transform(real_edge_map, best_M, output_shape=cad_edge_map.shape)
    
    # Compute alignment score
    score = _compute_alignment_score(aligned, cad_edge_map)
    
    # The score should be high (close to 1.0) since the images are identical
    assert score > 0.8, f"Expected high alignment score, got {score}"
    
    # Verify the matrix is close to identity (since no rotation is needed)
    identity = np.eye(3, dtype=np.float64)
    assert np.allclose(best_M, identity, atol=1e-6), \
        f"Expected identity matrix, got:\n{best_M}"


def test_resolve_180_ambiguity_with_rotated_shape():
    """
    Test that _resolve_180_ambiguity correctly handles a shape that is
    rotated 180° from the CAD reference.
    """
    # Create a CAD edge map with an L-shape
    cad_edge_map = create_simple_shape(size=200)
    
    # Create a real edge map that is rotated 180° from CAD
    center = (100, 100)
    rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
    real_edge_map = cv2.warpAffine(
        cad_edge_map,
        rotation_matrix,
        (200, 200),
        flags=cv2.INTER_NEAREST
    )
    
    # Define centroids (center of image)
    cad_centroid = (100.0, 100.0)
    real_centroid = (100.0, 100.0)
    
    # No scale difference
    scale = 1.0
    
    # PCA would estimate 0° but the correct angle is 180°
    pca_angle_deg = 0.0
    
    # Call _resolve_180_ambiguity
    best_M = _resolve_180_ambiguity(
        real_edge_map=real_edge_map,
        cad_edge_map=cad_edge_map,
        pca_angle_deg=pca_angle_deg,
        scale=scale,
        real_centroid=real_centroid,
        cad_centroid=cad_centroid,
    )
    
    # Apply the best transformation
    aligned = apply_transform(real_edge_map, best_M, output_shape=cad_edge_map.shape)
    
    # Compute alignment score
    score = _compute_alignment_score(aligned, cad_edge_map)
    
    # The score should be high since the correct 180° rotation should be selected
    assert score > 0.7, f"Expected high alignment score after 180° correction, got {score}"


def test_apply_transform_preserves_shape_and_dtype():
    """
    Test that apply_transform returns an array with the correct shape and dtype.
    """
    # Create a simple edge map
    edge_map = np.zeros((100, 100), dtype=np.uint8)
    edge_map[40:60, 40:60] = 255
    
    # Create an identity transformation matrix
    matrix = np.eye(3, dtype=np.float64)
    
    # Apply transform
    result = apply_transform(edge_map, matrix)
    
    # Check shape and dtype
    assert result.shape == edge_map.shape, \
        f"Expected shape {edge_map.shape}, got {result.shape}"
    assert result.dtype == np.uint8, \
        f"Expected dtype uint8, got {result.dtype}"


def test_apply_transform_with_custom_output_shape():
    """
    Test that apply_transform respects the output_shape parameter.
    """
    # Create a simple edge map
    edge_map = np.zeros((100, 100), dtype=np.uint8)
    edge_map[40:60, 40:60] = 255
    
    # Create an identity transformation matrix
    matrix = np.eye(3, dtype=np.float64)
    
    # Apply transform with custom output shape
    output_shape = (150, 150)
    result = apply_transform(edge_map, matrix, output_shape=output_shape)
    
    # Check shape
    assert result.shape == output_shape, \
        f"Expected shape {output_shape}, got {result.shape}"
    assert result.dtype == np.uint8, \
        f"Expected dtype uint8, got {result.dtype}"


def test_compute_alignment_score_bounds():
    """
    Test that _compute_alignment_score returns a value in [0.0, 1.0].
    """
    # Create two identical edge maps
    edge_map1 = np.zeros((100, 100), dtype=np.uint8)
    edge_map1[40:60, 40:60] = 255
    edge_map2 = edge_map1.copy()
    
    # Compute score for identical images
    score = _compute_alignment_score(edge_map1, edge_map2)
    assert 0.0 <= score <= 1.0, f"Score {score} is out of bounds [0.0, 1.0]"
    assert score > 0.9, f"Expected high score for identical images, got {score}"
    
    # Create two completely different edge maps
    edge_map3 = np.zeros((100, 100), dtype=np.uint8)
    edge_map3[10:30, 10:30] = 255
    edge_map4 = np.zeros((100, 100), dtype=np.uint8)
    edge_map4[70:90, 70:90] = 255
    
    # Compute score for non-overlapping images
    score2 = _compute_alignment_score(edge_map3, edge_map4)
    assert 0.0 <= score2 <= 1.0, f"Score {score2} is out of bounds [0.0, 1.0]"
    # Score should be low but might not be exactly 0 due to dilation
    assert score2 < 0.5, f"Expected low score for non-overlapping images, got {score2}"


def test_compute_alignment_score_empty_images():
    """
    Test that _compute_alignment_score handles empty images correctly.
    """
    # Create two empty edge maps
    edge_map1 = np.zeros((100, 100), dtype=np.uint8)
    edge_map2 = np.zeros((100, 100), dtype=np.uint8)
    
    # Compute score for empty images
    score = _compute_alignment_score(edge_map1, edge_map2)
    
    # Score should be 0.0 for empty images (union is 0)
    assert score == 0.0, f"Expected score 0.0 for empty images, got {score}"
