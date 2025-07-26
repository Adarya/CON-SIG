# CNA Signature Fitting Web App – Specification

This document describes **everything** the AI engine needs to generate a complete, runnable application that lets an end-user upload CNA segment or matrix files and returns:

1. A table of consensus‐signature activities (optionally with bootstrap‐based CIs).
2. A stacked-bar plot visualising signature contributions per sample.

The design re-uses the existing *CON_fitting* framework (already present in this repository) and adds a thin web interface.

---

## 1  High-level overview

• Frontend: **Streamlit** – single-page app, file uploader, parameter widgets, results area with tabs.
• Backend logic: Pure Python functions located in `CON_fitting_app/` (new) that wrap existing pipeline functions.
• Plotting: Matplotlib/Seaborn via `CON_fitting.src.visualizer.SignatureVisualizer`.
• Packaging: `requirements_app.txt` enumerating both scientific and web dependencies.
• Run command: `streamlit run app.py`.

Why Streamlit? It offers rapid prototyping, painless file upload handling, tight integration with Pandas, and automatic hot-reload.

---

## 2  Directory layout

```
CON_fitting_app/
 ├── app.py                     # Streamlit entry-point
 ├── backend.py                 # Thin wrapper around CON_fitting logic
 ├── plotting.py                # Helper focusing on stacked-bar plot for web
 ├── examples/
 │   ├── example_segments.seg   # Raw cbioportal file
 │   └── example_matrix.tsv     # Pre-prepared 28-feature matrix
 └── README.md                  # Quick-start for end-users
```

(The AI engine should generate **all** files above except those already existing in *CON_fitting*.)

---

## 3  Provided Python modules (already in repository, **do not modify**)

| File / module                                     | Purpose |
|---------------------------------------------------|---------|
| `CON_fitting/get_cna.py`                          | Convert cbioportal `.seg` → FACETS-like format |
| `CON_fitting/CNVMatrixGenerator.py`               | Build 28-feature CNV matrix |
| `CON_fitting/src/data_processor.py`               | Pre-processing and validation |
| `CON_fitting/src/signature_fitter.py`             | Core NNLS / EN fitting |
| `CON_fitting/src/visualizer.py`                   | Publication-quality plots (heatmap, stacked bar) |
| `CON_fitting_enhancements/bootstrapped_signature_fitter.py` | Optional uncertainty estimation |

The web app **imports** these modules – the AI engine only has to wire them together.

---

## 4  Detailed functional requirements

### 4.1 File input

1. **Upload widget** accepting either:
   - Raw cbioportal CNA segments (`*.seg` with `ID`, `chrom`, `loc.start`, `loc.end`, `seg.mean` columns), *or*
   - Pre-computed CNV28 matrix (`*.tsv`, `*.csv`).
2. If a `.seg` file is provided, call:
   ```python
   from CON_fitting.get_cna import process_cbioportal_to_facets
   from CON_fitting.CNVMatrixGenerator import generateCNVMatrix
   ```
   to obtain a matrix.
3. Display basic file statistics (rows, samples, features).

### 4.2 Parameter controls (sidebar)

| Widget | Default | Description |
|--------|---------|-------------|
| Checkbox | False | *Use bootstrap uncertainty* (activates `BootstrappedSignatureFitter`; else use `ConsensusSignatureFitter`). |
| Number input | 200 | *Bootstrap iterations (if enabled)* |
| Selectbox | `nnls` | Deconvolution method (`nnls`, `elastic_net`) |
| Button | *Run analysis* | Starts computation |

### 4.3 Computation pipeline

```text
User file → (optional) preprocess seg → 28-feature matrix →
DataProcessor.preprocess_cna_data() →
SignatureFitter / BootstrappedSignatureFitter.fit() →
Results cache (session_state)
```

### 4.4 Outputs

1. **Table:**
   • Activities dataframe rendered via `st.dataframe`, with download button (CSV).
   • If bootstrap enabled – additional columns `ci_lower`, `ci_upper` for each signature (multi-index columns preferred).
2. **Plot:**
   • Call `SignatureVisualizer.plot_signature_contributions_stacked()` to generate a Matplotlib Figure.
   • Render via `st.pyplot()`.
   • Download buttons for PNG & PDF versions.
3. **Quality metrics:**
   • Mean R², mean reconstruction error printed inline.

### 4.5 Performance & UX

• Use **Streamlit's `st.spinner()`** context while fitting.
• Persist results in `st.session_state` to avoid recomputation on app reload.
• Handle errors gracefully with `st.error()`.

---

## 5  Backend helper functions (`backend.py`)

```python
from pathlib import Path
import pandas as pd
from CON_fitting.get_cna import process_cbioportal_to_facets
from CON_fitting.CNVMatrixGenerator import generateCNVMatrix
from CON_fitting.src.data_processor import DataProcessor
from CON_fitting.src.signature_fitter import ConsensusSignatureFitter
from CON_fitting_enhancements.bootstrapped_signature_fitter import (
    BootstrappedSignatureFitter,
)

def load_user_file(uploaded_file: Path) -> pd.DataFrame:
    """Return CNA 28-feature matrix regardless of raw or pre-processed input."""
```

(… full code spec continues for AI engine – see *Implementation notes* below.)

---

## 6  Implementation notes for AI engine

1. **Environment**: Target Python ≥ 3.9.  Use `requirements_app.txt`:
   ```text
   streamlit>=1.35
   pandas
   numpy
   scipy
   scikit-learn
   matplotlib
   seaborn
   Pillow
   CON_fitting  # local path – already present in repo
   ```
2. **Path handling**: add project root to `sys.path` so that `CON_fitting` can be imported when running via Streamlit.
3. **Speed**: Bootstrapping can be CPU-heavy; use `n_iterations` slider to trade-off accuracy vs time.
4. **Styling**: Leverage Streamlit theme; include a collapsible sidebar with documentation / links.
5. **Packaging**: The AI engine should generate a `make run` or `run_app.sh` script for convenience.
6. **Testing**: Provide minimal `examples/` so app runs OOTB.

---

## 7  Example session (happy path)

1. User runs `streamlit run app.py`.
2. Browser opens at `localhost:8501`.
3. User uploads `data_cna_hg19.seg` (2 MB).
4. Checks *Use bootstrap uncertainty*, sets iterations = 500.
5. Clicks *Run analysis*.
6. Spinner shows *"Processing cbioportal seg → matrix (3 s)… Fitting signatures (4 s)… Bootstrapping 500 it (∼40 s)"*.
7. App displays:
   - Summary (samples = 67, mean R² = 0.93, error = 0.005).
   - Interactive table.
   - Stacked bar plot.
   - Download buttons (CSV, PNG, PDF).

---

## 8  Deliverables for AI engine

The engine must create **all new files** listed in §2 and ensure the app runs with:

```bash
pip install -r requirements_app.txt
streamlit run app.py
```

No further manual tweaks should be necessary.

---

*End of specification.* 