"""
Comparison and clustering module for handwriting analysis.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HandwritingComparator:
    """Compares handwriting samples and performs clustering."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize comparator.
        
        Args:
            similarity_threshold: Threshold for considering writings similar (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.scaler = StandardScaler()
    
    def compare_documents(
        self,
        features_by_doc: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compare all pairs of documents.
        
        Args:
            features_by_doc: Dictionary mapping document names to feature vectors
            
        Returns:
            Dictionary mapping document pairs to similarity scores
        """
        results = {}
        doc_names = list(features_by_doc.keys())
        
        logger.info(f"Comparing {len(doc_names)} documents...")
        
        for i, doc1 in enumerate(doc_names):
            for doc2 in doc_names[i + 1:]:
                similarity = self._compute_similarity(
                    features_by_doc[doc1],
                    features_by_doc[doc2]
                )
                results[(doc1, doc2)] = similarity
                
                if similarity > self.similarity_threshold:
                    logger.warning(
                        f"High similarity detected: {doc1} <-> {doc2} "
                        f"(score: {similarity:.3f})"
                    )
        
        return results
    
    def _compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        Compute similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Normalize features
        features1_norm = features1 / (np.linalg.norm(features1) + 1e-10)
        features2_norm = features2 / (np.linalg.norm(features2) + 1e-10)
        
        # Compute multiple distance metrics
        cosine_sim = 1 - cosine(features1_norm, features2_norm)
        
        # Euclidean similarity (converted to 0-1 scale)
        euclidean_dist = euclidean(features1_norm, features2_norm)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Correlation
        correlation = np.corrcoef(features1, features2)[0, 1]
        correlation_sim = (correlation + 1) / 2  # Scale to 0-1
        
        # Weighted combination
        similarity = (
            0.4 * cosine_sim +
            0.3 * euclidean_sim +
            0.3 * correlation_sim
        )
        
        return float(np.clip(similarity, 0, 1))
    
    def cluster_similar_writings(
        self,
        features_by_doc: Dict[str, np.ndarray],
        min_cluster_size: int = 2,
        method: str = 'agglomerative'
    ) -> List[Dict]:
        """
        Cluster documents with similar handwriting.
        
        Args:
            features_by_doc: Dictionary mapping document names to feature vectors
            min_cluster_size: Minimum number of documents to form a cluster
            method: Clustering method ('agglomerative' or 'dbscan')
            
        Returns:
            List of cluster dictionaries with document names and statistics
        """
        if len(features_by_doc) < 2:
            logger.warning("Need at least 2 documents for clustering")
            return []
        
        # Prepare data
        doc_names = list(features_by_doc.keys())
        features_matrix = np.array([features_by_doc[name] for name in doc_names])
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_matrix)
        
        # Perform clustering
        if method == 'dbscan':
            labels = self._cluster_dbscan(features_normalized)
        else:
            labels = self._cluster_agglomerative(features_normalized)
        
        # Process clusters
        clusters = self._process_clusters(
            labels, doc_names, features_matrix, min_cluster_size
        )
        
        logger.info(f"Found {len(clusters)} clusters with size >= {min_cluster_size}")
        
        return clusters
    
    def _cluster_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN algorithm."""
        # Convert similarity threshold to distance threshold
        eps = 1 - self.similarity_threshold
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=2,
            metric='euclidean'
        )
        return clustering.fit_predict(features)
    
    def _cluster_agglomerative(self, features: np.ndarray) -> np.ndarray:
        """Cluster using Agglomerative Hierarchical Clustering."""
        # Convert similarity threshold to distance threshold
        distance_threshold = 1 - self.similarity_threshold
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average'
        )
        return clustering.fit_predict(features)
    
    def _process_clusters(
        self,
        labels: np.ndarray,
        doc_names: List[str],
        features_matrix: np.ndarray,
        min_cluster_size: int
    ) -> List[Dict]:
        """
        Process clustering results into structured format.
        
        Args:
            labels: Cluster labels for each document
            doc_names: List of document names
            features_matrix: Matrix of feature vectors
            min_cluster_size: Minimum cluster size
            
        Returns:
            List of cluster information dictionaries
        """
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
            
            # Get documents in this cluster
            mask = labels == label
            cluster_docs = [doc_names[i] for i, m in enumerate(mask) if m]
            
            if len(cluster_docs) < min_cluster_size:
                continue
            
            # Compute cluster statistics
            cluster_features = features_matrix[mask]
            centroid = np.mean(cluster_features, axis=0)
            
            # Compute intra-cluster similarity
            similarities = []
            for i in range(len(cluster_features)):
                for j in range(i + 1, len(cluster_features)):
                    sim = self._compute_similarity(
                        cluster_features[i],
                        cluster_features[j]
                    )
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # Compute pairwise similarities
            pairwise_sims = {}
            for i, doc1 in enumerate(cluster_docs):
                for j, doc2 in enumerate(cluster_docs[i + 1:], start=i + 1):
                    sim = self._compute_similarity(
                        cluster_features[i],
                        cluster_features[j]
                    )
                    pairwise_sims[(doc1, doc2)] = sim
            
            cluster_info = {
                'cluster_id': int(label),
                'documents': cluster_docs,
                'size': len(cluster_docs),
                'avg_similarity': float(avg_similarity),
                'pairwise_similarities': pairwise_sims,
                'centroid': centroid
            }
            
            clusters.append(cluster_info)
        
        # Sort by average similarity (descending)
        clusters.sort(key=lambda x: x['avg_similarity'], reverse=True)
        
        return clusters
    
    def get_most_similar_pairs(
        self,
        results: Dict[Tuple[str, str], float],
        top_n: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Get the most similar document pairs.
        
        Args:
            results: Dictionary of pairwise similarities
            top_n: Number of top pairs to return
            
        Returns:
            List of (doc1, doc2, similarity) tuples
        """
        sorted_pairs = sorted(
            results.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            (pair[0], pair[1], score)
            for pair, score in sorted_pairs[:top_n]
        ]
    
    def compute_similarity_statistics(
        self,
        results: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """
        Compute statistical summary of similarities.
        
        Args:
            results: Dictionary of pairwise similarities
            
        Returns:
            Dictionary with statistical measures
        """
        if not results:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        similarities = list(results.values())
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities)),
            'q25': float(np.percentile(similarities, 25)),
            'q75': float(np.percentile(similarities, 75))
        }