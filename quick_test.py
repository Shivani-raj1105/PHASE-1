"""
Quick test script - uses the images you already have.

Just run: python quick_test.py
"""

import cv2
import numpy as np
from pathlib import Path
from cad_image_alignment import align


def main():
    print("=" * 70)
    print("Quick Alignment Test")
    print("=" * 70)
    
    # Look for common image names
    cad_candidates = ['cad.png', 'cad_front.png', 'cad_image.png', 'cad.jpg']
    real_candidates = ['real.png', 'real_photo.png', 'real_image.png', 'real.jpg', 'photo.jpg']
    
    cad_path = None
    real_path = None
    
    # Find CAD image
    for candidate in cad_candidates:
        if Path(candidate).exists():
            cad_path = candidate
            break
    
    # Find real image
    for candidate in real_candidates:
        if Path(candidate).exists():
            real_path = candidate
            break
    
    if cad_path is None or real_path is None:
        print("\n❌ Could not find images automatically.")
        print("\nPlease save your images as:")
        print("  - cad.png (or cad.jpg) - Your CAD front view image")
        print("  - real.png (or real.jpg) - Your camera photo")
        print("\nOr use: python test_real_images.py <cad_path> <real_path>")
        return
    
    print(f"\n📁 Found images:")
    print(f"   CAD:  {cad_path}")
    print(f"   Real: {real_path}")
    
    # Load images
    print(f"\n📥 Loading images...")
    cad = cv2.imread(cad_path, cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    
    if cad is None:
        print(f"❌ Could not load CAD image: {cad_path}")
        return
    
    if real is None:
        print(f"❌ Could not load real image: {real_path}")
        return
    
    print(f"   CAD shape:  {cad.shape}")
    print(f"   Real shape: {real.shape}")
    
    # Morphological gradient preprocessing - best match for CAD style
    print(f"\n🔧 Preprocessing...")
    
    # ========== CAD: Simple edge detection ==========
    _, cad_thresh = cv2.threshold(cad, 127, 255, cv2.THRESH_BINARY)
    cad_edges = cv2.Canny(cad_thresh, 50, 150)
    
    # ========== REAL: Morphological gradient (captures edges well) ==========
    print(f"   Processing real image (morphological gradient)...")
    
    # Step 1: Remove background
    blur = cv2.GaussianBlur(real, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Step 2: Clean up mask
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large, iterations=2)
    
    # Step 3: Apply mask to original image
    real_masked = cv2.bitwise_and(real, real, mask=mask)
    
    # Step 4: Apply morphological gradient to detect edges
    kernel_gradient = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(real_masked, cv2.MORPH_GRADIENT, kernel_gradient)
    
    # Step 5: Threshold the gradient
    _, real_edges = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
    
    # Step 6: Clean up with morphological closing
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    real_edges = cv2.morphologyEx(real_edges, cv2.MORPH_CLOSE, kernel_small)
    
    # Save debug images
    cv2.imwrite('debug_mask.png', mask)
    cv2.imwrite('debug_masked_image.png', real_masked)
    cv2.imwrite('debug_gradient.png', gradient)
    cv2.imwrite('debug_final_edges.png', real_edges)
    
    print(f"   Using Strategy: Morphological gradient")
    
    print(f"   CAD edges:  {np.count_nonzero(cad_edges)} pixels")
    print(f"   Real edges: {np.count_nonzero(real_edges)} pixels")
    
    # Run alignment
    print(f"\n⚙️  Running alignment...")
    result = align(cad_edges, real_edges)
    
    # Display results
    print(f"\n" + "=" * 70)
    print("✅ RESULTS")
    print("=" * 70)
    print(f"\n📊 Metrics:")
    print(f"   Strategy:        {result.strategy}")
    print(f"   Alignment Score: {result.alignment_score:.4f}")
    print(f"   Low Confidence:  {result.low_confidence}")
    if result.inlier_ratio:
        print(f"   Inlier Ratio:    {result.inlier_ratio:.2%}")
    
    # Save results
    print(f"\n💾 Saving results...")
    cv2.imwrite('quick_cad_edges.png', cad_edges)
    cv2.imwrite('quick_real_edges.png', real_edges)
    cv2.imwrite('quick_aligned.png', result.aligned_image)
    
    # Create simple overlay
    overlay = np.zeros((*cad_edges.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = cad_edges  # Red = CAD
    overlay[:, :, 1] = result.aligned_image  # Green = Aligned
    cv2.imwrite('quick_overlay.png', overlay)
    
    print(f"   ✓ quick_cad_edges.png")
    print(f"   ✓ quick_real_edges.png")
    print(f"   ✓ quick_aligned.png")
    print(f"   ✓ quick_overlay.png (Red=CAD, Green=Aligned, Yellow=Match)")
    
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
