"""
Test script for CAD-Image Alignment with real images.

This script demonstrates how to use the alignment module with your actual
CAD and camera images.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Import the alignment module
from cad_image_alignment import align, AlignmentResult


def preprocess_cad_image(image_path: str) -> np.ndarray:
    """
    Preprocess CAD image to produce binary edge map.
    
    For CAD images (clean line drawings):
    - Convert to grayscale
    - Apply Otsu thresholding
    - Apply Canny edge detection
    """
    print(f"📄 Loading CAD image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"   Original shape: {img.shape}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply Canny edge detection
    edges = cv2.Canny(thresh, 50, 150)
    
    print(f"   Edge pixels: {np.count_nonzero(edges)}")
    
    return edges


def preprocess_real_image(image_path: str, background_color: str = 'white') -> np.ndarray:
    """
    Preprocess real camera image to produce binary edge map.
    
    For real images (photos with background):
    - Remove background (green or white)
    - Convert to grayscale
    - Apply bilateral filtering (reduce reflections, preserve edges)
    - Apply Canny edge detection
    - Morphological cleaning (reduce noise)
    """
    print(f"📷 Loading real image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"   Original shape: {img.shape}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Background removal (simple thresholding for white/light backgrounds)
    if background_color == 'white':
        # Invert: background becomes black, object becomes white
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        # For other backgrounds, you might need color-based segmentation
        mask = gray
    
    # Apply bilateral filtering to reduce reflections while preserving edges
    filtered = cv2.bilateralFilter(mask, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply Canny edge detection
    edges = cv2.Canny(filtered, 50, 150)
    
    # Morphological cleaning to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    print(f"   Edge pixels: {np.count_nonzero(edges)}")
    
    return edges


def visualize_result(cad_edges: np.ndarray, real_edges: np.ndarray, 
                     result: AlignmentResult, output_path: str = 'alignment_result.png'):
    """
    Create a visualization showing:
    - Original CAD edges (red)
    - Original real edges (blue)
    - Aligned real edges (green)
    - Overlay of CAD and aligned (magenta = overlap)
    """
    print(f"\n🎨 Creating visualization...")
    
    # Create RGB canvas
    h, w = cad_edges.shape
    vis = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Top-left: CAD edges (red)
    vis[0:h, 0:w, 2] = cad_edges
    cv2.putText(vis, 'CAD (Reference)', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Top-right: Real edges (blue)
    vis[0:h, w:w*2, 0] = real_edges
    cv2.putText(vis, 'Real (Original)', (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Bottom-left: Aligned edges (green)
    vis[h:h*2, 0:w, 1] = result.aligned_image
    cv2.putText(vis, 'Aligned', (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Bottom-right: Overlay (CAD=red, Aligned=green, Overlap=yellow)
    vis[h:h*2, w:w*2, 2] = cad_edges  # Red channel = CAD
    vis[h:h*2, w:w*2, 1] = result.aligned_image  # Green channel = Aligned
    # Where both are present, you get yellow (red + green)
    
    # Add result info
    info_y = h + 60
    cv2.putText(vis, f'Strategy: {result.strategy}', (w + 10, info_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, f'Score: {result.alignment_score:.4f}', (w + 10, info_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, f'Low Conf: {result.low_confidence}', (w + 10, info_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if result.inlier_ratio is not None:
        cv2.putText(vis, f'Inliers: {result.inlier_ratio:.2%}', (w + 10, info_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(vis, 'Overlay (Yellow=Match)', (w + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Save visualization
    cv2.imwrite(output_path, vis)
    print(f"   Saved to: {output_path}")
    
    return vis


def main():
    """
    Main function to test alignment with real images.
    """
    print("=" * 70)
    print("CAD-Image Alignment Test with Real Images")
    print("=" * 70)
    
    # Check if image paths are provided
    if len(sys.argv) < 3:
        print("\n❌ Usage: python test_real_images.py <cad_image_path> <real_image_path>")
        print("\nExample:")
        print("  python test_real_images.py cad_front.png real_photo.jpg")
        print("\nOr place your images in the current directory and run:")
        print("  python test_real_images.py cad.png real.png")
        return
    
    cad_path = sys.argv[1]
    real_path = sys.argv[2]
    
    # Check if files exist
    if not Path(cad_path).exists():
        print(f"\n❌ CAD image not found: {cad_path}")
        return
    
    if not Path(real_path).exists():
        print(f"\n❌ Real image not found: {real_path}")
        return
    
    try:
        # Step 1: Preprocess images
        print("\n" + "=" * 70)
        print("STEP 1: Preprocessing")
        print("=" * 70)
        
        cad_edges = preprocess_cad_image(cad_path)
        real_edges = preprocess_real_image(real_path, background_color='white')
        
        # Save preprocessed images for inspection
        cv2.imwrite('preprocessed_cad.png', cad_edges)
        cv2.imwrite('preprocessed_real.png', real_edges)
        print(f"\n💾 Saved preprocessed images:")
        print(f"   - preprocessed_cad.png")
        print(f"   - preprocessed_real.png")
        
        # Step 2: Run alignment
        print("\n" + "=" * 70)
        print("STEP 2: Running Alignment")
        print("=" * 70)
        
        result = align(cad_edges, real_edges)
        
        # Step 3: Display results
        print("\n" + "=" * 70)
        print("STEP 3: Results")
        print("=" * 70)
        
        print(f"\n✅ Alignment Complete!")
        print(f"\n📊 Results:")
        print(f"   Strategy:        {result.strategy}")
        print(f"   Alignment Score: {result.alignment_score:.4f}")
        print(f"   Low Confidence:  {result.low_confidence}")
        if result.inlier_ratio is not None:
            print(f"   Inlier Ratio:    {result.inlier_ratio:.2%}")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if result.strategy == "homography":
            print(f"   ✓ Fine alignment succeeded (best quality)")
        elif result.strategy == "affine_coarse_only":
            print(f"   ⚠ Fine alignment failed, using coarse only")
        else:
            print(f"   ⚠ Both alignments failed, no transformation applied")
        
        if result.alignment_score >= 0.7:
            print(f"   ✓ Good alignment quality (score ≥ 0.7)")
        elif result.alignment_score >= 0.5:
            print(f"   ⚠ Moderate alignment quality (0.5 ≤ score < 0.7)")
        elif result.alignment_score >= 0.3:
            print(f"   ⚠ Poor alignment quality (0.3 ≤ score < 0.5)")
        else:
            print(f"   ❌ Very poor alignment (score < 0.3)")
        
        # Step 4: Save results
        print("\n" + "=" * 70)
        print("STEP 4: Saving Results")
        print("=" * 70)
        
        # Save aligned image
        cv2.imwrite('aligned_result.png', result.aligned_image)
        print(f"\n💾 Saved aligned image: aligned_result.png")
        
        # Save transformation matrix
        np.save('transform_matrix.npy', result.transform_matrix)
        print(f"💾 Saved transformation matrix: transform_matrix.npy")
        
        # Create and save visualization
        vis = visualize_result(cad_edges, real_edges, result)
        
        print("\n" + "=" * 70)
        print("✅ Done! Check the output files:")
        print("=" * 70)
        print("   1. preprocessed_cad.png     - CAD edge map")
        print("   2. preprocessed_real.png    - Real edge map")
        print("   3. aligned_result.png       - Aligned real image")
        print("   4. alignment_result.png     - Visualization (4-panel)")
        print("   5. transform_matrix.npy     - Transformation matrix")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
