# handwriting-plagiarism-detector

Compares the handwriting in scanned PDF exercise sheets and ranks the pairs and groups of documents that may have been written by the same hand, for a teacher to review.

[![CI](https://github.com/Fede-Berga/handwriting-plagiarism-detector/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Fede-Berga/handwriting-plagiarism-detector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

It was written to screen handwritten physics exercises handed in as scans. If one person writes several students' sheets, those pages should share stroke, slant and spacing statistics; the tool measures them and points at the documents worth comparing by eye.

## Quick start

Prerequisites:

- Python 3.11 or newer (tested with 3.13)
- [uv](https://docs.astral.sh/uv/) (tested with 0.13.6)
- Poppler, which pdf2image calls to render pages (tested with 26.10): `brew install poppler` on macOS, `sudo apt-get install poppler-utils` on Debian and Ubuntu

```sh
git clone https://github.com/Fede-Berga/handwriting-plagiarism-detector.git
cd handwriting-plagiarism-detector
uv sync --python 3.13 --extra dev
```

Put one PDF per submission in `data/input/`, then run:

```sh
uv run python -m handwriting_detector.main data/input --output data/output
```

Options: `--threshold`/`-t` sets the similarity above which a pair is flagged (default 0.85), `--dpi` the rendering resolution (default 300), `--output`/`-o` the output folder (default `data/output`), and `--verbose`/`-v` enables debug logging. Flagged pairs are logged as warnings, and the output folder receives:

- `similarity_matrix.png`: heatmap of every pairwise score
- `suspicious_pairs.csv`: the ten highest-scoring pairs, whatever their score
- `clusters.png`: size and internal similarity of each group, only when a group was found
- `report.html`: summary statistics, top pairs and groups (the page text is in Italian)

Documents are named after their PDF files. The same pipeline is available from Python:

```python
from handwriting_detector import HandwritingDetector

detector = HandwritingDetector(dpi=300, similarity_threshold=0.85)
results = detector.analyze_folder("data/input", "data/output")
results["top_pairs"]  # [(doc_a, doc_b, score), ...], highest first
results["clusters"]   # [{"documents": [...], "avg_similarity": ..., ...}, ...]
```

## How it works

1. **Rendering and clean-up** ([`pdf_processor.py`](src/handwriting_detector/pdf_processor.py)): pdf2image renders each page in greyscale at the chosen DPI; OpenCV applies a bilateral filter, CLAHE contrast enhancement, adaptive thresholding, non-local-means denoising and a morphological opening.
2. **Features** ([`feature_extractor.py`](src/handwriting_detector/feature_extractor.py)): 15 statistics per page, averaged over the document: stroke width (distance transform), slant (ellipses fitted to connected components), gaps in the column and row projection profiles, ink density, component aspect ratio, edge-based smoothness, contour curvature (mean and standard deviation), horizontal, vertical and diagonal stroke density, the intensity variance of dark pixels, and the skewness and kurtosis of all pixel intensities.
3. **Comparison** ([`comparator.py`](src/handwriting_detector/comparator.py)): every pair gets a score in [0, 1], computed as 0.4 × cosine similarity + 0.3 × 1/(1 + Euclidean distance) of the L2-normalised vectors + 0.3 × their Pearson correlation rescaled to [0, 1]. Separately, agglomerative clustering (average linkage) on standardised features, with a distance cut-off of 1 − threshold, forms the groups; groups of two or more documents are reported.
4. **Report** ([`visualizer.py`](src/handwriting_detector/visualizer.py)): writes the files listed above.

## Running the tests

```sh
uv run pytest
```

The four unit tests check the feature-vector length, binarisation, the score of two close vectors and the package version. They use synthetic arrays, so they need neither Poppler nor scans. [CI](.github/workflows/ci.yml) runs the same commands on Python 3.13.

## Student data

Scanned exercises and the reports derived from them are personal data. `data/input/` and `data/output/` are gitignored apart from their `.gitkeep` files, so scans and results kept there stay out of version control. Document names, taken from the file names, appear in the log, the heatmap, the CSV and the HTML report: use pseudonyms if a report has to leave your machine.

## Limitations

- It ranks similarity; it does not identify writers or prove copying. A high score is a reason to compare the sheets side by side, never evidence on its own, and the HTML report carries the same warning.
- The features are page-level statistics. Besides the writer, they depend on scan quality, resolution, pen, paper ruling and the amount and layout of the written content.
- Scores are not calibrated. The pairwise score uses unscaled features, so a few large-valued ones (projection gaps in pixels, slant in degrees, intensity kurtosis) carry most of the weight: in a check with four synthetic typed pages, two different typefaces scored 0.96, above the default threshold.
- Clustering uses a different scale (Euclidean distance between standardised features), so a pair above the threshold need not end up in a group; the two typefaces above did not.
- The repository contains no labelled data or accuracy evaluation.
- `HandwritingDetector.find_plagiarism_groups()` is a placeholder that returns an empty list; use the result of `analyze_folder()`.

## License

[MIT](LICENSE)
