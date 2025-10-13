"""
Main entry point for the handwriting plagiarism detector.
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

from .pdf_processor import PDFProcessor
from .feature_extractor import FeatureExtractor
from .comparator import HandwritingComparator
from .visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HandwritingDetector:
    """Main class orchestrating the plagiarism detection pipeline."""
    
    def __init__(
        self,
        dpi: int = 300,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the detector.
        
        Args:
            dpi: Resolution for PDF conversion
            similarity_threshold: Threshold for similarity detection (0-1)
        """
        self.pdf_processor = PDFProcessor(dpi=dpi)
        self.feature_extractor = FeatureExtractor()
        self.comparator = HandwritingComparator(similarity_threshold=similarity_threshold)
        self.visualizer = Visualizer()
        
        logger.info(f"HandwritingDetector initialized (DPI={dpi}, threshold={similarity_threshold})")
    
    def analyze_folder(
        self,
        input_folder: Path,
        output_folder: Optional[Path] = None
    ) -> dict:
        """
        Analyze all PDFs in a folder.
        
        Args:
            input_folder: Path to folder containing PDF files
            output_folder: Path to save results (optional)
            
        Returns:
            Dictionary with analysis results
        """
        input_folder = Path(input_folder)
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        # Find all PDFs
        pdf_files = list(input_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_folder}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Process PDFs
        images_by_doc = self.pdf_processor.process_pdfs(pdf_files)
        
        if not images_by_doc:
            logger.error("Failed to process any PDFs")
            return {}
        
        # Extract features
        logger.info("Extracting features...")
        features_by_doc = {}
        for doc_name, images in images_by_doc.items():
            features = self.feature_extractor.extract_features_from_images(images)
            features_by_doc[doc_name] = features
            logger.info(f"Extracted features from {doc_name}")
        
        # Compare documents
        logger.info("Comparing documents...")
        similarities = self.comparator.compare_documents(features_by_doc)
        
        # Find clusters
        logger.info("Clustering similar documents...")
        clusters = self.comparator.cluster_similar_writings(features_by_doc)
        
        # Get statistics
        stats = self.comparator.compute_similarity_statistics(similarities)
        top_pairs = self.comparator.get_most_similar_pairs(similarities, top_n=10)
        
        results = {
            'features': features_by_doc,
            'similarities': similarities,
            'clusters': clusters,
            'statistics': stats,
            'top_pairs': top_pairs,
            'num_documents': len(features_by_doc)
        }
        
        # Generate visualizations if output folder specified
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            self.generate_report(results, output_folder)
        
        return results
    
    def generate_report(self, results: dict, output_folder: Path):
        """
        Generate visualization and reports.
        
        Args:
            results: Analysis results dictionary
            output_folder: Path to save reports
        """
        logger.info(f"Generating reports in {output_folder}")
        
        # Save similarity matrix
        self.visualizer.plot_similarity_matrix(
            results['similarities'],
            output_folder / 'similarity_matrix.png'
        )
        
        # Save clusters visualization
        if results['clusters']:
            self.visualizer.plot_clusters(
                results['clusters'],
                output_folder / 'clusters.png'
            )
        
        # Generate HTML report
        self.visualizer.generate_html_report(
            results,
            output_folder / 'report.html'
        )
        
        # Save suspicious pairs CSV
        self.visualizer.save_suspicious_pairs_csv(
            results['top_pairs'],
            output_folder / 'suspicious_pairs.csv'
        )
        
        logger.info("Report generation complete")
    
    def find_plagiarism_groups(
        self,
        similarity_threshold: Optional[float] = None,
        min_group_size: int = 2
    ) -> list:
        """
        Find groups of potentially plagiarized documents.
        
        Args:
            similarity_threshold: Override default threshold
            min_group_size: Minimum number of documents in a group
            
        Returns:
            List of suspicious groups
        """
        if similarity_threshold:
            self.comparator.similarity_threshold = similarity_threshold
        
        # This assumes features have been extracted
        # In practice, you'd call analyze_folder first
        logger.info("Finding plagiarism groups...")
        return []


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Detect potential plagiarism in handwritten documents'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Folder containing PDF files to analyze'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='data/output',
        help='Output folder for results (default: data/output)'
    )
    parser.add_argument(
        '--threshold',
        '-t',
        type=float,
        default=0.85,
        help='Similarity threshold (0-1, default: 0.85)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF conversion (default: 300)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize detector
    detector = HandwritingDetector(
        dpi=args.dpi,
        similarity_threshold=args.threshold
    )
    
    # Run analysis
    try:
        results = detector.analyze_folder(
            Path(args.input_folder),
            Path(args.output)
        )
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Documents analyzed: {results['num_documents']}")
        logger.info(f"Clusters found: {len(results['clusters'])}")
        logger.info(f"Average similarity: {results['statistics']['mean']:.3f}")
        logger.info(f"Results saved to: {args.output}")
        
        if results['top_pairs']:
            logger.info("\nTop suspicious pairs:")
            for doc1, doc2, score in results['top_pairs'][:5]:
                logger.info(f"  {doc1} <-> {doc2}: {score:.3f}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())