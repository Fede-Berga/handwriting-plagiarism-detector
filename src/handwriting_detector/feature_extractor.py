"""
Feature extraction module for handwriting analysis.
"""
import logging
from typing import List, Tuple

import cv2
import numpy as np
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts features from handwriting images."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
            'avg_stroke_width', 'avg_slant', 'avg_spacing',
            'pressure_variance', 'aspect_ratio', 'line_spacing',
            'character_density', 'stroke_smoothness',
            'curvature_mean', 'curvature_std',
            'horizontal_density', 'vertical_density',
            'diagonal_density', 'skewness', 'kurtosis_val'
        ]
    
    def extract_features_from_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple images (pages).
        
        Args:
            images: List of grayscale images
            
        Returns:
            Feature vector (averaged across all pages)
        """
        all_features = []
        
        for idx, image in enumerate(images):
            try:
                features = self._extract_single_image_features(image)
                all_features.append(features)
            except Exception as e:
                logger.warning(f"Error extracting features from page {idx}: {e}")
                continue
        
        if not all_features:
            logger.error("No features extracted from any page")
            return np.zeros(len(self.feature_names))
        
        # Average features across all pages
        return np.mean(all_features, axis=0)
    
    def _extract_single_image_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Feature vector
        """
        # Preprocess
        binary = self._binarize(image)
        
        # Extract various features
        features = []
        
        # 1. Stroke width
        features.append(self._compute_stroke_width(binary))
        
        # 2. Slant angle
        features.append(self._compute_slant(binary))
        
        # 3. Character spacing
        features.append(self._compute_spacing(binary))
        
        # 4. Pressure variance (grayscale intensity)
        features.append(self._compute_pressure_variance(image))
        
        # 5. Aspect ratio
        features.append(self._compute_aspect_ratio(binary))
        
        # 6. Line spacing
        features.append(self._compute_line_spacing(binary))
        
        # 7. Character density
        features.append(self._compute_character_density(binary))
        
        # 8. Stroke smoothness
        features.append(self._compute_stroke_smoothness(binary))
        
        # 9-10. Curvature statistics
        curv_mean, curv_std = self._compute_curvature_stats(binary)
        features.extend([curv_mean, curv_std])
        
        # 11-13. Directional densities
        h_dens, v_dens, d_dens = self._compute_directional_densities(binary)
        features.extend([h_dens, v_dens, d_dens])
        
        # 14-15. Statistical moments
        features.append(self._compute_skewness(image))
        features.append(self._compute_kurtosis(image))
        
        return np.array(features)
    
    @staticmethod
    def _binarize(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        return binary
    
    def _compute_stroke_width(self, binary: np.ndarray) -> float:
        """Compute average stroke width using morphological operations."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        stroke_pixels = cv2.subtract(binary, eroded)
        
        if np.sum(stroke_pixels) == 0:
            return 0.0
        
        # Distance transform to estimate width
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        avg_width = np.mean(dist[dist > 0]) if np.any(dist > 0) else 0.0
        
        return float(avg_width)
    
    def _compute_slant(self, binary: np.ndarray) -> float:
        """Estimate slant angle of handwriting."""
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        angles = []
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angles.append(ellipse[2])
        
        return float(np.mean(angles)) if angles else 0.0
    
    def _compute_spacing(self, binary: np.ndarray) -> float:
        """Compute average horizontal spacing between characters."""
        horizontal_projection = np.sum(binary, axis=0)
        
        # Find gaps (valleys in projection)
        threshold = np.mean(horizontal_projection) * 0.1
        gaps = horizontal_projection < threshold
        
        # Calculate gap widths
        gap_widths = []
        in_gap = False
        gap_start = 0
        
        for i, is_gap in enumerate(gaps):
            if is_gap and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_gap and in_gap:
                gap_widths.append(i - gap_start)
                in_gap = False
        
        return float(np.mean(gap_widths)) if gap_widths else 0.0
    
    def _compute_pressure_variance(self, image: np.ndarray) -> float:
        """Compute variance in grayscale intensity (pressure proxy)."""
        # Focus on darker pixels (writing)
        writing_pixels = image[image < 200]
        return float(np.var(writing_pixels)) if len(writing_pixels) > 0 else 0.0
    
    def _compute_aspect_ratio(self, binary: np.ndarray) -> float:
        """Compute average aspect ratio of connected components."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        ratios = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 0:
                ratios.append(w / h)
        
        return float(np.mean(ratios)) if ratios else 1.0
    
    def _compute_line_spacing(self, binary: np.ndarray) -> float:
        """Compute average vertical spacing between lines."""
        vertical_projection = np.sum(binary, axis=1)
        
        # Find line gaps
        threshold = np.mean(vertical_projection) * 0.1
        gaps = vertical_projection < threshold
        
        gap_heights = []
        in_gap = False
        gap_start = 0
        
        for i, is_gap in enumerate(gaps):
            if is_gap and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_gap and in_gap:
                gap_heights.append(i - gap_start)
                in_gap = False
        
        return float(np.mean(gap_heights)) if gap_heights else 0.0
    
    def _compute_character_density(self, binary: np.ndarray) -> float:
        """Compute density of writing on page."""
        total_pixels = binary.size
        writing_pixels = np.sum(binary > 0)
        return float(writing_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    def _compute_stroke_smoothness(self, binary: np.ndarray) -> float:
        """Compute smoothness of strokes using edge detection."""
        edges = cv2.Canny(binary, 50, 150)
        
        # Smoothness is inversely related to edge density
        edge_density = np.sum(edges > 0) / edges.size
        return float(1.0 - edge_density)
    
    def _compute_curvature_stats(self, binary: np.ndarray) -> Tuple[float, float]:
        """Compute curvature statistics of writing."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        curvatures = []
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Approximate curvature at each point
            for i in range(1, len(contour) - 1):
                p1 = contour[i - 1][0]
                p2 = contour[i][0]
                p3 = contour[i + 1][0]
                
                # Angle between consecutive segments
                v1 = p2 - p1
                v2 = p3 - p2
                
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                curvatures.append(abs(angle))
        
        if curvatures:
            return float(np.mean(curvatures)), float(np.std(curvatures))
        return 0.0, 0.0
    
    def _compute_directional_densities(self, binary: np.ndarray) -> Tuple[float, float, float]:
        """Compute densities in different directions."""
        # Horizontal strokes
        kernel_h = np.array([[1, 1, 1]], dtype=np.uint8)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        h_density = np.sum(horizontal > 0) / binary.size
        
        # Vertical strokes
        kernel_v = np.array([[1], [1], [1]], dtype=np.uint8)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
        v_density = np.sum(vertical > 0) / binary.size
        
        # Diagonal strokes
        kernel_d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        diagonal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_d)
        d_density = np.sum(diagonal > 0) / binary.size
        
        return float(h_density), float(v_density), float(d_density)
    
    def _compute_skewness(self, image: np.ndarray) -> float:
        """Compute skewness of pixel intensity distribution."""
        pixels = image.flatten()
        return float(skew(pixels))
    
    def _compute_kurtosis(self, image: np.ndarray) -> float:
        """Compute kurtosis of pixel intensity distribution."""
        pixels = image.flatten()
        return float(kurtosis(pixels))