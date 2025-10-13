"""
Visualization module for displaying analysis results.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class Visualizer:
    """Generates visualizations and reports for plagiarism detection results."""
    
    def __init__(self):
        """Initialize visualizer with style settings."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        logger.info("Visualizer initialized")
    
    def plot_similarity_matrix(
        self,
        similarities: Dict[Tuple[str, str], float],
        output_path: Path,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Create a heatmap of document similarities.
        
        Args:
            similarities: Dictionary of pairwise similarities
            output_path: Path to save the plot
            figsize: Figure size (width, height)
        """
        if not similarities:
            logger.warning("No similarities to plot")
            return
        
        # Extract unique document names
        doc_names = sorted(set(
            [doc for pair in similarities.keys() for doc in pair]
        ))
        
        # Create similarity matrix
        n = len(doc_names)
        matrix = np.zeros((n, n))
        
        for (doc1, doc2), score in similarities.items():
            i = doc_names.index(doc1)
            j = doc_names.index(doc2)
            matrix[i, j] = score
            matrix[j, i] = score
        
        # Set diagonal to 1.0 (self-similarity)
        np.fill_diagonal(matrix, 1.0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            xticklabels=doc_names,
            yticklabels=doc_names,
            square=True,
            cbar_kws={'label': 'Similarity Score'},
            ax=ax
        )
        
        ax.set_title('Document Similarity Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity matrix saved to {output_path}")
    
    def plot_clusters(
        self,
        clusters: List[Dict],
        output_path: Path,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Visualize detected clusters.
        
        Args:
            clusters: List of cluster dictionaries
            output_path: Path to save the plot
            figsize: Figure size
        """
        if not clusters:
            logger.warning("No clusters to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Cluster sizes
        cluster_ids = [c['cluster_id'] for c in clusters]
        cluster_sizes = [c['size'] for c in clusters]
        
        bars = ax1.bar(cluster_ids, cluster_sizes, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Cluster ID', fontweight='bold')
        ax1.set_ylabel('Number of Documents', fontweight='bold')
        ax1.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        # Plot 2: Average similarities within clusters
        avg_similarities = [c['avg_similarity'] for c in clusters]
        
        bars = ax2.barh(cluster_ids, avg_similarities, color='coral', alpha=0.7)
        ax2.set_ylabel('Cluster ID', fontweight='bold')
        ax2.set_xlabel('Average Similarity', fontweight='bold')
        ax2.set_title('Cluster Internal Similarity', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Annotate bars
        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width,
                bar.get_y() + bar.get_height() / 2.,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster visualization saved to {output_path}")
    
    def plot_similarity_distribution(
        self,
        similarities: Dict[Tuple[str, str], float],
        output_path: Path,
        threshold: float = 0.85
    ):
        """
        Plot distribution of similarity scores.
        
        Args:
            similarities: Dictionary of pairwise similarities
            output_path: Path to save the plot
            threshold: Highlight values above this threshold
        """
        if not similarities:
            return
        
        scores = list(similarities.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(
            scores,
            bins=30,
            color='skyblue',
            edgecolor='black',
            alpha=0.7
        )
        
        # Color bars above threshold in red
        for i, patch in enumerate(patches):
            if bins[i] >= threshold:
                patch.set_facecolor('red')
                patch.set_alpha(0.8)
        
        # Add vertical line at threshold
        ax.axvline(
            threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold})'
        )
        
        ax.set_xlabel('Similarity Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Similarity Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution plot saved to {output_path}")
    
    def plot_feature_comparison(
        self,
        features_by_doc: Dict[str, np.ndarray],
        feature_names: List[str],
        output_path: Path,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create radar/spider plot comparing features across documents.
        
        Args:
            features_by_doc: Dictionary mapping document names to feature vectors
            feature_names: Names of features
            output_path: Path to save the plot
            figsize: Figure size
        """
        if not features_by_doc or len(features_by_doc) > 5:
            logger.warning("Feature comparison works best with 2-5 documents")
            return
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Normalize features for better visualization
        all_features = np.array(list(features_by_doc.values()))
        normalized_features = (all_features - all_features.min(axis=0)) / (
            all_features.max(axis=0) - all_features.min(axis=0) + 1e-10
        )
        
        # Setup angles for radar chart
        num_features = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each document
        colors = plt.cm.Set3(np.linspace(0, 1, len(features_by_doc)))
        for idx, (doc_name, features) in enumerate(features_by_doc.items()):
            values = normalized_features[idx].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=doc_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, size=8)
        ax.set_ylim(0, 1)
        ax.set_title('Feature Comparison Across Documents', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature comparison plot saved to {output_path}")
    
    def save_suspicious_pairs_csv(
        self,
        top_pairs: List[Tuple[str, str, float]],
        output_path: Path
    ):
        """
        Save suspicious pairs to CSV file.
        
        Args:
            top_pairs: List of (doc1, doc2, similarity) tuples
            output_path: Path to save CSV
        """
        if not top_pairs:
            logger.warning("No suspicious pairs to save")
            return
        
        df = pd.DataFrame(
            top_pairs,
            columns=['Document 1', 'Document 2', 'Similarity Score']
        )
        
        df.to_csv(output_path, index=False)
        logger.info(f"Suspicious pairs saved to {output_path}")
    
    def generate_html_report(
        self,
        results: dict,
        output_path: Path
    ):
        """
        Generate comprehensive HTML report.
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save HTML report
        """
        html_content = self._build_html_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def _build_html_report(self, results: dict) -> str:
        """Build HTML report content."""
        stats = results.get('statistics', {})
        clusters = results.get('clusters', [])
        top_pairs = results.get('top_pairs', [])
        
        html = f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report di Rilevamento Plagio Grafia</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            display: flex;
            align-items: center;
        }}
        h2::before {{
            content: '';
            width: 5px;
            height: 30px;
            background: #667eea;
            margin-right: 15px;
            border-radius: 3px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 14px;
            opacity: 0.95;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 42px;
            font-weight: bold;
            margin-top: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 16px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 13px;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .warning {{
            background: linear-gradient(to right, #fff3cd, #fff8e1);
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 30px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(255, 193, 7, 0.2);
        }}
        .warning strong {{
            color: #f57c00;
            font-size: 1.1em;
        }}
        .alert {{
            background: linear-gradient(to right, #f8d7da, #ffe0e0);
            border-left: 5px solid #dc3545;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(220, 53, 69, 0.2);
        }}
        .alert h3 {{
            color: #dc3545;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .alert p {{
            margin: 8px 0;
            line-height: 1.6;
        }}
        .score {{
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            font-size: 0.95em;
        }}
        .score-high {{
            background-color: #dc3545;
            color: white;
            box-shadow: 0 2px 6px rgba(220, 53, 69, 0.4);
        }}
        .score-medium {{
            background-color: #ffc107;
            color: #333;
            box-shadow: 0 2px 6px rgba(255, 193, 7, 0.4);
        }}
        .score-low {{
            background-color: #28a745;
            color: white;
            box-shadow: 0 2px 6px rgba(40, 167, 69, 0.4);
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
        }}
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-style: italic;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
            .stat-card:hover {{
                transform: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 Report di Rilevamento Plagio Grafia</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Documenti Analizzati</h3>
                <div class="value">{results.get('num_documents', 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Cluster Trovati</h3>
                <div class="value">{len(clusters)}</div>
            </div>
            <div class="stat-card">
                <h3>Similarità Media</h3>
                <div class="value">{stats.get('mean', 0):.3f}</div>
            </div>
            <div class="stat-card">
                <h3>Similarità Massima</h3>
                <div class="value">{stats.get('max', 0):.3f}</div>
            </div>
        </div>
        
        <h2>📊 Riepilogo Statistico</h2>
        <table>
            <tr>
                <th>Metrica</th>
                <th>Valore</th>
            </tr>
            <tr>
                <td>Similarità Media</td>
                <td>{stats.get('mean', 0):.4f}</td>
            </tr>
            <tr>
                <td>Deviazione Standard</td>
                <td>{stats.get('std', 0):.4f}</td>
            </tr>
            <tr>
                <td>Minimo</td>
                <td>{stats.get('min', 0):.4f}</td>
            </tr>
            <tr>
                <td>25° Percentile</td>
                <td>{stats.get('q25', 0):.4f}</td>
            </tr>
            <tr>
                <td>Mediana</td>
                <td>{stats.get('median', 0):.4f}</td>
            </tr>
            <tr>
                <td>75° Percentile</td>
                <td>{stats.get('q75', 0):.4f}</td>
            </tr>
            <tr>
                <td>Massimo</td>
                <td>{stats.get('max', 0):.4f}</td>
            </tr>
        </table>
        
        <h2>⚠️ Coppie Più Sospette</h2>
        {self._build_suspicious_pairs_table(top_pairs)}
        
        <h2>🔍 Cluster Rilevati</h2>
        {self._build_clusters_section(clusters)}
        
        <div class="warning">
            <strong>⚠️ Nota Importante:</strong> Questi risultati sono basati su un'analisi automatizzata 
            e non dovrebbero essere l'unica base per accuse di plagio. È necessaria una revisione umana 
            e prove aggiuntive per qualsiasi determinazione formale.
        </div>
        
        <div class="timestamp">
            Report generato automaticamente dal sistema di rilevamento plagio grafia
        </div>
    </div>
</body>
</html>
        """
        return html
    
    def _build_suspicious_pairs_table(self, top_pairs: List[Tuple[str, str, float]]) -> str:
        """Build HTML table for suspicious pairs."""
        if not top_pairs:
            return '<div class="no-data">Nessuna coppia sospetta rilevata.</div>'
        
        rows = []
        for doc1, doc2, score in top_pairs:
            score_class = self._get_score_class(score)
            rows.append(f"""
            <tr>
                <td>{doc1}</td>
                <td>{doc2}</td>
                <td><span class="score {score_class}">{score:.4f}</span></td>
            </tr>
            """)
        
        return f"""
        <table>
            <tr>
                <th>Documento 1</th>
                <th>Documento 2</th>
                <th>Punteggio Similarità</th>
            </tr>
            {''.join(rows)}
        </table>
        """
    
    def _build_clusters_section(self, clusters: List[Dict]) -> str:
        """Build HTML section for clusters."""
        if not clusters:
            return '<div class="no-data">Nessun cluster rilevato.</div>'
        
        sections = []
        for cluster in clusters:
            docs = ', '.join(cluster['documents'])
            sections.append(f"""
            <div class="alert">
                <h3>Cluster {cluster['cluster_id']}</h3>
                <p><strong>Documenti:</strong> {docs}</p>
                <p><strong>Dimensione:</strong> {cluster['size']} documenti</p>
                <p><strong>Similarità Media:</strong> <span class="score {self._get_score_class(cluster['avg_similarity'])}">{cluster['avg_similarity']:.4f}</span></p>
            </div>
            """)
        
        return ''.join(sections)
    
    @staticmethod
    def _get_score_class(score: float) -> str:
        """Get CSS class based on score."""
        if score >= 0.85:
            return 'score-high'
        elif score >= 0.7:
            return 'score-medium'
        else:
            return 'score-low'