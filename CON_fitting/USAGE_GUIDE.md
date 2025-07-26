# CNA Signature Analysis - Quick Start Guide

## 🚀 One-Click Pipeline: cbioportal → Signatures

Transform raw cbioportal CNA segments into signature activities in one command:

```bash
# Navigate to CON_fitting directory
cd CON_fitting

# Basic usage
python cbioportal_to_signatures.py input_file.seg ProjectName

# Example with rectal cancer data
python cbioportal_to_signatures.py /path/to/data_cna_hg19.seg RectalMSK2022
```

**Output**: Complete analysis results in `results/ProjectName_TIMESTAMP/` including:
- Processed segments and CNV matrix
- Fitted signature activities  
- Quality metrics and visualizations
- Comprehensive analysis report

---

## 🏥 Clinical Analysis: Signatures → Survival

Perform Cox regression and Kaplan-Meier survival analysis with flexible input files:

```bash
# From the consensus directory, run clinical analysis with file arguments
cd clinical_analysis_fitting

# Basic usage with full paths
python run_clinical_analysis.py \
    --sig_file ../CON_fitting/results/RectalMSK2022_Enhanced_20250607_163110/signatures/RectalMSK2022_Enhanced_signature_activities.csv \
    --clinical_file /path/to/survival_indexed_msk-rectal.csv

# Using short options
python run_clinical_analysis.py \
    -s ../CON_fitting/results/MyProject_*/signatures/*_signature_activities.csv \
    -c /path/to/clinical_data.csv

# Help and usage information
python run_clinical_analysis.py --help
```

**Required File Formats**:
- **Signature file**: CSV with samples as rows, signatures as columns (e.g., `consensus_1`, `consensus_2`, etc.)
- **Clinical file**: CSV with `SAMPLE_ID`, `OS_MONTHS`, and `OS_STATUS` columns

**Output**: Clinical results including:
- Cox regression hazard ratios (`cox_analysis/cox_results.csv`)
- Kaplan-Meier survival curves (`kaplan_meier/km_*.png`)
- Forest plots (`visualizations/cox_forest_plot.png`)
- Comprehensive reports (`reports/clinical_analysis_report.txt`)

---

## 📊 Complete Workflow Example

```bash
# Step 1: Generate signature activities
cd CON_fitting
python cbioportal_to_signatures.py data_cna_hg19.seg MyProject

# Step 2: Run clinical analysis (update paths as needed)
cd ../clinical_analysis_fitting
python run_clinical_analysis.py \
    --sig_file ../CON_fitting/results/MyProject_*/signatures/*_signature_activities.csv \
    --clinical_file /path/to/your_clinical_data.csv

# Check results
ls -la ../CON_fitting/clinical_analysis_results_*/
```

**Directory Structure**:
```
consensus/
├── CON_fitting/                           # Signature analysis framework
│   ├── cbioportal_to_signatures.py        # One-click pipeline
│   ├── results/ProjectName_TIMESTAMP/     # Signature fitting results
│   └── clinical_analysis_results_*/       # Clinical analysis outputs
└── clinical_analysis_fitting/             # Clinical analysis scripts
    └── run_clinical_analysis.py           # Clinical survival analysis
```

**Key Features**:
- ⚡ **Lightning fast**: Complete analysis in seconds
- 📈 **Publication-ready**: High-quality plots with Arial fonts
- 🔬 **Comprehensive**: From raw data to clinical insights
- 📁 **Organized**: Results saved with timestamps for version control
- 🔧 **Flexible**: Accepts any signature and clinical data files

**Dependencies**: Ensure you have `lifelines` installed:
```bash
pip install lifelines
``` 