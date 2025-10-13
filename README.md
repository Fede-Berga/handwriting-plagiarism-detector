# 📝 Handwriting Plagiarism Detector

Sistema per rilevare possibili plagi analizzando la grafia in PDF manoscritti di esercizi di fisica.

## 🏗️ Struttura del Progetto

```
handwriting-plagiarism-detector/
├── pyproject.toml              # Configurazione uv e dipendenze
├── README.md                   # Questo file
├── .gitignore                  # File da ignorare in git
├── .python-version             # Versione Python (3.11)
├── src/
│   └── handwriting_detector/
│       ├── __init__.py
│       ├── main.py            # Entry point principale
│       ├── pdf_processor.py   # Gestione PDF
│       ├── feature_extractor.py  # Estrazione features
│       ├── comparator.py      # Confronto e clustering
│       └── visualizer.py      # Visualizzazione risultati
├── tests/
│   ├── __init__.py
│   └── test_detector.py       # Test unitari
├── data/
│   ├── input/                 # PDF da analizzare (gitignored)
│   └── output/                # Risultati (gitignored)
└── notebooks/
    └── analysis.ipynb         # Jupyter notebook per analisi

```

## 🚀 Installazione

### Prerequisiti
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) installato
- Poppler (per pdf2image):
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`
  - Windows: Scarica da [poppler windows](https://github.com/oschwartz10612/poppler-windows/releases/)

### Setup

```bash
# Clona il repository
git clone https://github.com/Fede-Berga/handwriting-plagiarism-detector
cd handwriting-plagiarism-detector

# Installa dipendenze con uv
uv sync

# Attiva l'ambiente virtuale
source .venv/bin/activate  # Linux/macOS
# oppure
.venv\Scripts\activate  # Windows
```

## 📊 Utilizzo

### 1. Preparazione Dati
Inserisci i PDF degli studenti nella cartella `data/input/`:
```
data/input/
├── studente_mario_rossi.pdf
├── studente_lucia_verdi.pdf
└── ...
```

### 2. Analisi Base
```bash
# Esegui l'analisi completa
uv run python -m handwriting_detector.main

# Con opzioni personalizzate
uv run python -m handwriting_detector.main --threshold 0.85 --dpi 300
```

### 3. Analisi Avanzata
```python
from handwriting_detector import HandwritingDetector

# Inizializza il detector
detector = HandwritingDetector()

# Analizza tutti i PDF
results = detector.analyze_folder("data/input")

# Trova gruppi sospetti
suspicious_groups = detector.find_plagiarism_groups(
    similarity_threshold=0.85,
    min_group_size=2
)

# Visualizza report
detector.generate_report("data/output/report.html")
```

## 🔧 Configurazione

Modifica `pyproject.toml` per personalizzare:
- Soglia di similarità
- Parametri DPI per conversione PDF
- Algoritmi di clustering

## 📈 Output

Il sistema genera:
- **report.html**: Report interattivo con grafici
- **similarity_matrix.png**: Matrice di similarità tra tutti i documenti
- **clusters.json**: Gruppi identificati con score di similarità
- **suspicious_pairs.csv**: Coppie sospette ordinate per similarità

## 🧪 Testing

```bash
# Esegui tutti i test
uv run pytest

# Con coverage
uv run pytest --cov=handwriting_detector

# Test specifici
uv run pytest tests/test_detector.py::test_feature_extraction
```

## 🛠️ Tecnologie Utilizzate

- **OpenCV**: Elaborazione immagini
- **scikit-learn**: Machine learning e clustering
- **pdf2image**: Conversione PDF
- **NumPy**: Calcoli numerici
- **Matplotlib/Seaborn**: Visualizzazioni
- **uv**: Gestione dipendenze e ambiente

## 📋 Algoritmo

1. **Conversione**: PDF → Immagini (alta risoluzione)
2. **Preprocessing**: Binarizzazione, rimozione rumore
3. **Feature Extraction**:
   - Densità inchiostro
   - Inclinazione scrittura
   - Spaziatura caratteri
   - Distribuzione tratti
   - Pattern ricorrenti
4. **Comparazione**: Calcolo similarità coseno tra features
5. **Clustering**: DBSCAN per identificare gruppi
6. **Report**: Generazione risultati e visualizzazioni

## ⚠️ Limitazioni

- Richiede scansioni di qualità decente
- Funziona meglio con scrittura in corsivo
- Non sostituisce la valutazione umana
- Può avere falsi positivi con grafie simili naturalmente

## 📄 Licenza

MIT License

## 👥 Contributi

Contributi benvenuti! Apri una issue o pull request.

## 📞 Supporto

Per problemi o domande, apri una issue su GitHub.
