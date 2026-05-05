"""
Unit tests for _validate_inputs function.
"""

import logging
import numpy as np
import pytest

from cad_image_alignment.alignment import _validate_inputs


def test_valid_inputs():
    """Test that valid inputs pass validation."""
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Should not raise any exception and return the real edge map
    result = _validate_inputs(cad, real)
    assert result.shape == cad.shape
    assert result.dtype == np.uint8


def test_invalid_dtype_cad():
    """Test that invalid dtype for CAD edge map raises ValueError."""
    cad = np.ones((100, 100), dtype=np.float32)
    real = np.ones((100, 100), dtype=np.uint8) * 255
    
    with pytest.raises(ValueError, match="cad_edge_map has invalid dtype"):
        _validate_inputs(cad, real)


def test_invalid_dtype_real():
    """Test that invalid dtype for real edge map raises ValueError."""
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.ones((100, 100), dtype=np.int32)
    
    with pytest.raises(ValueError, match="real_edge_map has invalid dtype"):
        _validate_inputs(cad, real)


def test_invalid_ndim_cad():
    """Test that invalid ndim for CAD edge map raises ValueError."""
    cad = np.ones((100, 100, 3), dtype=np.uint8)
    real = np.ones((100, 100), dtype=np.uint8) * 255
    
    with pytest.raises(ValueError, match="cad_edge_map has invalid ndim"):
        _validate_inputs(cad, real)


def test_invalid_ndim_real():
    """Test that invalid ndim for real edge map raises ValueError."""
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.ones((100,), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="real_edge_map has invalid ndim"):
        _validate_inputs(cad, real)


def test_empty_cad():
    """Test that empty CAD edge map raises ValueError."""
    cad = np.zeros((100, 100), dtype=np.uint8)
    real = np.ones((100, 100), dtype=np.uint8) * 255
    
    with pytest.raises(ValueError, match="cad_edge_map is empty"):
        _validate_inputs(cad, real)


def test_empty_real():
    """Test that empty real edge map raises ValueError."""
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.zeros((100, 100), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="real_edge_map is empty"):
        _validate_inputs(cad, real)


def test_both_invalid():
    """Test that first invalid input is reported when both are invalid."""
    cad = np.zeros((100, 100), dtype=np.float32)
    real = np.zeros((100, 100), dtype=np.int32)
    
    # Should report CAD dtype error first
    with pytest.raises(ValueError, match="cad_edge_map has invalid dtype"):
        _validate_inputs(cad, real)


def test_resolution_mismatch_resize():
    """Test that resolution mismatch triggers resize of real_edge_map."""
    # Create CAD edge map with shape (100, 100)
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Create real edge map with different shape (50, 50)
    real = np.ones((50, 50), dtype=np.uint8) * 255
    
    # Validate and get resized real edge map
    result = _validate_inputs(cad, real)
    
    # Result should have same shape as CAD edge map
    assert result.shape == cad.shape
    assert result.dtype == np.uint8
    
    # Result should be resized version of real edge map
    assert result.shape == (100, 100)


def test_resolution_mismatch_logs_debug(caplog):
    """Test that resolution mismatch emits DEBUG-level log message."""
    # Create CAD edge map with shape (100, 100)
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Create real edge map with different shape (50, 50)
    real = np.ones((50, 50), dtype=np.uint8) * 255
    
    # Capture logs at DEBUG level
    with caplog.at_level(logging.DEBUG, logger='cad_image_alignment.alignment'):
        result = _validate_inputs(cad, real)
    
    # Check that DEBUG log message was emitted
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'DEBUG'
    assert 'Resolution mismatch detected' in caplog.records[0].message
    assert '(50, 50)' in caplog.records[0].message
    assert '(100, 100)' in caplog.records[0].message
