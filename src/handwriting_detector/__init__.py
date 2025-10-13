"""
Handwriting Plagiarism Detector

A system for detecting potential plagiarism by analyzing handwriting in scanned PDF documents.
Uses computer vision and machine learning to extract features and identify similar writing patterns.
"""

__version__ = "0.1.0"
__author__ = "Federico Bergamini"
__license__ = "MIT"

from .pdf_processor import PDFProcessor
from .feature_extractor import FeatureExtractor
from .comparator import HandwritingComparator
from .visualizer import Visualizer
from .main import HandwritingDetector

__all__ = [
    "PDFProcessor",
    "FeatureExtractor",
    "HandwritingComparator",
    "Visualizer",
    "HandwritingDetector",
]

# Package metadata
METADATA = {
    "name": "handwriting-detector",
    "version": __version__,
    "description": "Handwriting plagiarism detector using computer vision and clustering",
    "author": __author__,
    "license": __license__,
}