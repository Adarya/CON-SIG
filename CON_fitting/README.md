# CON_fitting: Consensus CNA Signature Fitting Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust framework for fitting consensus Copy Number Alteration (CNA) signatures to new sample data, developed for reproducible research and clinical applications.

## Overview

CON_fitting implements multiple deconvolution algorithms to accurately determine consensus signature activities in new CNA samples. The framework provides comprehensive validation, visualization, and quality assessment tools for reliable signature analysis.

## Features

- **Multiple Fitting Methods**: NNLS, Elastic Net, and Constrained Least Squares
- **Comprehensive Validation**: Statistical metrics, correlation analysis, and distribution comparisons
- **Rich Visualization**: Heatmaps, quality metrics plots, and correlation analyses
- **Robust Data Processing**: Input validation, preprocessing, and format standardization
- **Production Ready**: Logging, error handling, and comprehensive documentation

## Installation

### Requirements

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Setup

1. Clone or download the CON_fitting framework
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.data_processor import DataProcessor
from src.signature_fitter import ConsensusSignatureFitter
from src.visualizer import SignatureVisualizer

# Initialize components
processor = DataProcessor()
fitter = ConsensusSignatureFitter(
    consensus_signatures=consensus_sigs,
    method='nnls'
)
visualizer = SignatureVisualizer()

# Load and process data
cna_data = processor.load_cna_data('your_cna_data.csv')
processed_data = processor.preprocess_cna_data(cna_data)

# Fit signatures
activities, metrics = fitter.fit(processed_data)

# Visualize results
figures = visualizer.create_summary_report(
    activities=activities,
    metrics=metrics,
    output_dir='output/'
)
```

### Validation Test

Run the included validation test with 100 random samples:

```bash
cd CON_fitting
python test_validation.py
```

This will:
- Test all fitting methods
- Generate comprehensive validation metrics
- Create visualization plots
- Save results to `output/` directory

## Framework Components

### 1. DataProcessor (`src/data_processor.py`)

Handles data loading, validation, and preprocessing:

- **Input formats**: CSV, TSV, Excel
- **Validation**: Data quality checks and issue reporting
- **Preprocessing**: Missing value handling, normalization, alignment
- **Utilities**: Random subset creation, data summarization

### 2. ConsensusSignatureFitter (`src/signature_fitter.py`)

Core signature fitting functionality:

- **Methods**: 
  - NNLS (Non-Negative Least Squares)
  - Elastic Net with non-negativity constraints
  - Constrained Least Squares optimization
- **Quality metrics**: R², reconstruction error, confidence intervals
- **Method comparison**: Automated benchmarking across algorithms

### 3. SignatureVisualizer (`src/visualizer.py`)

Comprehensive visualization capabilities:

- **Activity heatmaps**: Sample×signature activity matrices
- **Distribution plots**: Box plots and violin plots of activities
- **Correlation analysis**: Fitted vs reference comparisons
- **Quality metrics**: R² distributions, error analysis
- **Method comparison**: Performance benchmarking plots

### 4. SignatureValidator (`src/validator.py`)

Validation and quality assessment:

- **Correlation metrics**: Pearson and Spearman correlations
- **Error metrics**: MSE, MAE, RMSE
- **Distribution analysis**: KS tests, Wasserstein distance
- **Rank correlation**: Sample and signature ranking analysis
- **Quality scoring**: Composite quality assessment

## Input Data Format

### CNA Data
- **Format**: CSV file with samples as rows, CNA categories as columns
- **Index**: Sample identifiers
- **Columns**: CNA category names (must match consensus signatures)
- **Values**: Non-negative numeric values

Example:
```
Sample,0:homdel:0-100kb,0:homdel:100kb-1Mb,...
Sample1,0.1,0.05,...
Sample2,0.2,0.1,...
```

### Consensus Signatures
- **Format**: CSV file with CNA categories as rows, signatures as columns
- **Index**: CNA category names
- **Columns**: Signature identifiers (e.g., consensus_1, consensus_2, ...)
- **Values**: Normalized signature weights (should sum to 1)

## Output Files

### Generated Files
- `fitted_activities.csv`: Fitted signature activities
- `fitting_metrics.csv`: Per-sample quality metrics
- `validation_report.txt`: Detailed validation results
- `summary_report.md`: Executive summary in Markdown
- `*.png`: Visualization plots

### Quality Metrics
- **R² Score**: Proportion of variance explained
- **Reconstruction Error**: Mean squared error between input and reconstructed data
- **Correlation**: Pearson correlation with reference activities
- **Quality Score**: Composite metric (0-1 scale)

## Validation Results Interpretation

### Quality Thresholds
- **Excellent**: R² > 0.8, Correlation > 0.9
- **Good**: R² > 0.7, Correlation > 0.8
- **Moderate**: R² > 0.5, Correlation > 0.6
- **Poor**: Below moderate thresholds

### Success Criteria
- Overall correlation > 0.8
- Mean R² > 0.7
- At least 4/5 signatures with correlation > 0.8
- Quality score > 0.8

## Advanced Usage

### Custom Fitting Parameters

```python
# Initialize with custom parameters
fitter = ConsensusSignatureFitter(
    consensus_signatures=signatures,
    method='elastic_net',
    normalize=True,
    verbose=True
)

# Method-specific parameters can be adjusted in the source code
```

### Method Comparison

```python
# Compare all methods on the same data
comparison_results = fitter.compare_methods(
    cna_data, 
    methods=['nnls', 'elastic_net', 'constrained_ls']
)
```

### Comprehensive Validation

```python
# Full validation against reference data
validator = SignatureValidator()
validation_results = validator.validate_activities(
    fitted_activities=fitted_activities,
    reference_activities=reference_activities
)
```

## Troubleshooting

### Common Issues

1. **Missing CNA categories**: Framework will add missing categories with zeros
2. **Negative values**: Automatically clipped to zero with warning
3. **No common samples**: Check sample naming consistency
4. **Poor fitting quality**: Try different methods or check data quality

### Error Messages

- `No common CNA categories found`: Check column naming between input and signatures
- `No common samples found`: Verify sample identifiers match reference data
- `Optimization failed`: Data may have quality issues or extreme values

## Contributing

To contribute to CON_fitting:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use CON_fitting in your research, please cite:

```
CON_fitting: A robust framework for consensus CNA signature fitting
[Your publication details here]
```

## Support

For questions, issues, or feature requests:
- Create an issue in the repository
- Contact the development team
- Check the documentation and examples

## Changelog

### Version 1.0.0
- Initial release
- Core fitting algorithms (NNLS, Elastic Net, Constrained LS)
- Comprehensive validation framework
- Visualization and reporting tools
- Production-ready deployment features 