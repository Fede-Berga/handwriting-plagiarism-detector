"""
PDF processing module for converting PDFs to images.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF to image conversion and preprocessing."""
    
    def __init__(self, dpi: int = 300):
        """
        Initialize PDF processor.
        
        Args:
            dpi: Resolution for PDF conversion (default: 300)
        """
        self.dpi = dpi
        logger.info(f"PDFProcessor initialized with DPI={dpi}")
    
    def process_pdf(self, pdf_path: Path) -> List[np.ndarray]:
        """
        Convert a single PDF to a list of images (one per page).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of images as numpy arrays (grayscale)
        """
        try:
            logger.info(f"Converting {pdf_path.name} to images...")
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                grayscale=True,
                fmt='png'
            )
            
            # Convert PIL images to numpy arrays and preprocess
            numpy_images = []
            for idx, img in enumerate(images):
                # Convert to numpy array
                img_array = np.array(img)
                
                # Preprocess image
                preprocessed = self.preprocess_image(img_array)
                numpy_images.append(preprocessed)
                
                logger.debug(f"Processed page {idx + 1}/{len(images)} from {pdf_path.name}")
            
            logger.info(f"Successfully converted {len(numpy_images)} pages from {pdf_path.name}")
            return numpy_images
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            return []
    
    def process_pdfs(self, pdf_paths: List[Path]) -> Dict[str, List[np.ndarray]]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping document names to lists of page images
        """
        results = {}
        
        logger.info(f"Processing {len(pdf_paths)} PDF files...")
        
        for idx, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"Processing file {idx}/{len(pdf_paths)}: {pdf_path.name}")
            
            doc_name = pdf_path.stem  # Filename without extension
            images = self.process_pdf(pdf_path)
            
            if images:
                results[doc_name] = images
            else:
                logger.warning(f"No images extracted from {pdf_path.name}")
        
        logger.info(f"Successfully processed {len(results)}/{len(pdf_paths)} PDF files")
        return results
    
    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better feature extraction.
        
        Applies:
        - Noise reduction
        - Contrast enhancement
        - Binarization
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)
        
        # Apply adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Denoise the binary image
        denoised = cv2.fastNlMeansDenoising(binary, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Remove small noise using morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return cleaned
    
    @staticmethod
    def detect_text_regions(image: np.ndarray) -> List[tuple]:
        """
        Detect text regions in the image using contour detection.
        
        Args:
            image: Binary image
            
        Returns:
            List of bounding boxes (x, y, w, h) for text regions
        """
        # Find contours
        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    @staticmethod
    def remove_borders(image: np.ndarray, border_size: int = 50) -> np.ndarray:
        """
        Remove borders from the image.
        
        Args:
            image: Input image
            border_size: Size of border to remove (pixels)
            
        Returns:
            Image with borders removed
        """
        h, w = image.shape[:2]
        
        # Check if image is large enough
        if h <= 2 * border_size or w <= 2 * border_size:
            logger.warning("Image too small to remove borders")
            return image
        
        return image[border_size:h-border_size, border_size:w-border_size]
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """
        Deskew (straighten) a skewed image.
        
        Args:
            image: Input binary image
            
        Returns:
            Deskewed image
        """
        # Find all non-zero points
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 10:
            logger.warning("Not enough points for deskewing")
            return image
        
        # Find minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        # Only deskew if angle is significant
        if abs(angle) < 0.5:
            return image
        
        # Get image center and rotation matrix
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug(f"Image deskewed by {angle:.2f} degrees")
        return rotated
    
    def save_processed_images(
        self,
        images: List[np.ndarray],
        output_dir: Path,
        prefix: str = "page"
    ):
        """
        Save processed images to disk.
        
        Args:
            images: List of images to save
            output_dir: Directory to save images
            prefix: Prefix for filenames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, image in enumerate(images):
            filename = output_dir / f"{prefix}_{idx + 1:03d}.png"
            cv2.imwrite(str(filename), image)
            logger.debug(f"Saved {filename}")
        
        logger.info(f"Saved {len(images)} images to {output_dir}")
    
    def extract_page_from_pdf(
        self,
        pdf_path: Path,
        page_number: int
    ) -> Optional[np.ndarray]:
        """
        Extract a single page from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number to extract (1-indexed)
            
        Returns:
            Image as numpy array, or None if extraction fails
        """
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_number,
                last_page=page_number,
                grayscale=True
            )
            
            if images:
                img_array = np.array(images[0])
                return self.preprocess_image(img_array)
            else:
                logger.error(f"Failed to extract page {page_number} from {pdf_path.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting page {page_number} from {pdf_path.name}: {e}")
            return None
    
    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """
        Get the number of pages in a PDF without full conversion.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages, or 0 if error
        """
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(pdf_path)
            return info.get('Pages', 0)
        except Exception as e:
            logger.error(f"Error getting page count for {pdf_path.name}: {e}")
            return 0