# CONSIG - CNA Signature Analysis Web Application

[![License: Academic](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io/)

🧬 **CONSIG** (CON_fitting Signature Interface for Genomics) is a user-friendly web application for CNA (Copy Number Alteration) signature analysis. It provides an intuitive interface for analyzing genomic data and extracting meaningful signature patterns using consensus CNA signatures.

**✅ Self-Contained**: Includes the complete CON_fitting framework - no external dependencies needed!  
**✅ Import Issues Fixed**: All CON_fitting modules are included and ready to use!

## ✨ Key Features

- **📁 Multiple Input Formats**: Support for cbioportal `.seg` files and pre-processed CNA matrices
- **🔬 Advanced Analysis**: NNLS and Elastic Net deconvolution methods
- **📊 Bootstrap Uncertainty**: Confidence interval estimation through resampling
- **📈 Interactive Visualizations**: Publication-ready plots with export options
- **🚀 Easy Deployment**: Single-command launch for local use
- **🐳 Docker Support**: Containerized deployment for production environments

## Features

- 📁 **File Upload Support**: Upload cbioportal `.seg` files or pre-processed CNA matrices
- 🔬 **Advanced Analysis**: Signature deconvolution using NNLS or Elastic Net methods
- 📊 **Interactive Visualizations**: Stacked bar plots, heatmaps, and quality metrics
- 🎯 **Bootstrap Uncertainty**: Optional confidence interval estimation
- 📥 **Export Results**: Download tables (CSV) and plots (PNG/PDF)
- 🚀 **Easy Deployment**: Both local and cloud deployment options

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Git (for cloning the repository)

### Installation & Launch

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd consensus/CON_fitting_app
   ```

2. **Launch the application**:
   ```bash
   ./run_app.sh
   ```

   The script will automatically:
   - Create a virtual environment
   - Install all dependencies
   - Launch the Streamlit application
   - Open your browser to `http://localhost:8501`

### Alternative Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_app.txt

# Run the application
streamlit run app.py
```

## Usage Guide

### 1. File Upload

The application accepts two types of input files:

#### cbioportal Segment Files (`.seg`)
- **Format**: Tab-separated values with columns: `ID`, `chrom`, `loc.start`, `loc.end`, `seg.mean`
- **Example**: `examples/example_segments.seg`
- **Processing**: Automatically converted to 28-feature CNA matrix using the CON_fitting pipeline

#### Pre-processed Matrices (`.tsv`, `.csv`)
- **Format**: Samples as rows, CNA features as columns
- **Example**: `examples/example_matrix.tsv`
- **Requirements**: 28 CNA features (columns) for optimal results

### 2. Analysis Parameters

Configure your analysis using the sidebar controls:

- **Bootstrap Uncertainty**: Enable for confidence intervals (slower but more robust)
- **Bootstrap Iterations**: Number of resampling iterations (50-1000)
- **Deconvolution Method**: 
  - `nnls`: Non-negative least squares (default)
  - `elastic_net`: Elastic net regularization

### 3. Results Interpretation

The application provides:

#### Quality Metrics
- **Mean R²**: Goodness of fit across all samples
- **Mean Reconstruction Error**: Average error in signature reconstruction

#### Signature Activities Table
- Sample-by-signature activity matrix
- Optional confidence intervals (if bootstrap enabled)
- Download as CSV for further analysis

#### Visualizations
- **Stacked Bar Plot**: Signature contributions per sample
- **Interactive Display**: Zoom, pan, and explore results
- **Export Options**: PNG and PDF formats

## File Structure

```
CON_fitting_app/
├── app.py                       # Main Streamlit application
├── backend.py                   # Core processing functions
├── plotting.py                  # Visualization utilities
├── consensus_signatures.csv     # Reference consensus signatures
├── requirements_app.txt         # Python dependencies
├── run_app.sh                   # Launch script
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Docker Compose setup
├── examples/                    # Sample data files
│   ├── example_segments.seg     # Sample cbioportal file
│   └── example_matrix.tsv       # Sample CNA matrix
├── test_backend.py              # Backend functionality tests
├── test_consensus_signatures.py # Consensus signatures tests
├── test_full_backend.py         # Full workflow tests
├── INSTALLATION.md              # Installation guide
└── README.md                    # This file
```

## Advanced Deployment

### Docker Deployment

For containerized deployment:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t consig-app .
docker run -p 8501:8501 consig-app
```

### Cloud Deployment

The application is designed to be cloud-ready:

1. **Heroku**: Use the included `Dockerfile`
2. **AWS/GCP**: Deploy using container services
3. **Azure**: Use Azure Container Instances

### Environment Variables

Configure the application using these environment variables:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)
- `STREAMLIT_SERVER_HEADLESS`: Run without browser (default: false)

## Technical Details

### Dependencies

The application integrates with existing CON_fitting modules:

- `CON_fitting.get_cna`: Convert cbioportal files to FACETS format
- `CON_fitting.CNVMatrixGenerator`: Generate 28-feature CNA matrices
- `CON_fitting.src.data_processor`: Data preprocessing and validation
- `CON_fitting.src.signature_fitter`: Core signature fitting algorithms
- `CON_fitting_enhancements.bootstrapped_signature_fitter`: Uncertainty estimation

### Performance Considerations

- **File Size**: Optimized for files up to 100MB
- **Sample Count**: Efficiently handles 1-1000 samples
- **Bootstrap**: CPU-intensive; adjust iterations based on requirements
- **Memory**: Scales with sample count and feature dimensionality

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure CON_fitting modules are in the project directory
2. **Port Conflicts**: Change port using `--port` flag: `./run_app.sh --port 8502`
3. **Memory Issues**: Reduce bootstrap iterations or process smaller files
4. **File Format**: Verify input files match expected formats

### Error Messages

- **"Invalid file format"**: Check file extension and content structure
- **"Missing required columns"**: Ensure `.seg` files have required columns
- **"Matrix contains missing values"**: Clean data before upload

### Getting Help

If you encounter issues:

1. Check the console output for detailed error messages
2. Verify input file formats match the specifications
3. Ensure all dependencies are properly installed
4. Try the provided example files first

## Development

### Contributing

To contribute to CONSIG:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Testing

Run the application with example data:

```bash
# Start the application
./run_app.sh

# Upload examples/example_matrix.tsv
# Configure parameters and run analysis
# Verify results are generated correctly
```

### Code Structure

- `app.py`: Streamlit UI and user interactions
- `backend.py`: Business logic and CON_fitting integration
- `plotting.py`: Visualization and export functions

## 📜 License

This software is licensed for **academic use only**. See [LICENSE](LICENSE) for full terms.

- ✅ Academic research and educational use
- ✅ Non-commercial scientific investigations  
- ✅ Scholarly publications and presentations
- ❌ Commercial use without permission

## 📖 Citation

If you use CONSIG in your research, please cite:

```bibtex
@software{consig2025,
  title={CONSIG: CNA Signature Analysis Web Application},
  author={[Author Names]},
  year={2025},
  url={https://github.com/Adarya/CON-SIG},
  note={Academic software for CNA signature analysis}
}
```

## 🆘 Support

For support and questions:

- 🐛 **Issues**: [GitHub Issues](https://github.com/Adarya/CON-SIG/issues)
- 📧 **Contact**: [Maintainer email]
- 📖 **Documentation**: See files in this repository

---

**CONSIG** - Making CNA signature analysis accessible to everyone! 🧬✨