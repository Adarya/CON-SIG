"""
Backend functions for the CONSIG web application.

This module provides wrapper functions that integrate the existing CON_fitting framework
with the Streamlit web interface, handling file processing, validation, and signature fitting.
"""

import sys
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
from typing import Dict, Any, Optional, Tuple
import streamlit as st

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import existing CON_fitting modules
    from CON_fitting.get_cna import process_cbioportal_to_facets
    from CON_fitting.CNVMatrixGenerator import generateCNVMatrix
    from CON_fitting.src.data_processor import DataProcessor
    from CON_fitting.src.signature_fitter import ConsensusSignatureFitter
    from CON_fitting_enhancements.bootstrapped_signature_fitter import BootstrappedSignatureFitter
except ImportError as e:
    st.error(f"Failed to import CON_fitting modules: {e}")
    st.error("Please ensure the CON_fitting framework is properly installed.")
    sys.exit(1)

def validate_file_format(uploaded_file) -> bool:
    """
    Validate that the uploaded file has the correct format.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        bool: True if file format is valid
    """
    if uploaded_file is None:
        return False
        
    file_extension = Path(uploaded_file.name).suffix.lower()
    valid_extensions = ['.seg', '.tsv', '.csv']
    
    return file_extension in valid_extensions

def get_file_statistics(uploaded_file) -> Dict[str, Any]:
    """
    Get basic statistics about the uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dict with file statistics
    """
    if not validate_file_format(uploaded_file):
        raise ValueError("Invalid file format. Please upload a .seg, .tsv, or .csv file.")
    
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Reset file pointer to beginning
    uploaded_file.seek(0)
    
    try:
        if file_extension == '.seg':
            # Read seg file
            df = pd.read_csv(uploaded_file, sep='\t')
            
            # Basic validation for seg format
            required_columns = ['ID', 'chrom', 'loc.start', 'loc.end', 'seg.mean']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in .seg file: {missing_columns}")
            
            n_samples = df['ID'].nunique()
            n_segments = len(df)
            
            return {
                'file_type': 'seg',
                'n_samples': n_samples,
                'n_features': n_segments,
                'file_size': len(uploaded_file.getvalue()),
                'columns': list(df.columns)
            }
            
        else:
            # Read matrix file
            separator = '\t' if file_extension == '.tsv' else ','
            df = pd.read_csv(uploaded_file, sep=separator, index_col=0)
            
            return {
                'file_type': file_extension.lstrip('.'),
                'n_samples': len(df),
                'n_features': len(df.columns),
                'file_size': len(uploaded_file.getvalue()),
                'columns': list(df.columns)
            }
            
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")
    
    finally:
        # Reset file pointer for subsequent reads
        uploaded_file.seek(0)

def load_user_file(uploaded_file) -> pd.DataFrame:
    """
    Load and process user file, returning a standardized CNA matrix.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame: Processed CNA matrix with samples as rows and features as columns
    """
    if not validate_file_format(uploaded_file):
        raise ValueError("Invalid file format")
    
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    try:
        if file_extension == '.seg':
            # Process cbioportal seg file
            return _process_seg_file(uploaded_file)
        else:
            # Process pre-computed matrix
            return _process_matrix_file(uploaded_file, file_extension)
            
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

def _process_seg_file(uploaded_file) -> pd.DataFrame:
    """
    Process a cbioportal .seg file into a CNA matrix.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame: Processed CNA matrix
    """
    import tempfile
    import os
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.seg', delete=False) as tmp_seg_file:
        tmp_seg_file.write(uploaded_file.getvalue())
        tmp_seg_path = tmp_seg_file.name
    
    # Create temporary output directory
    temp_output_dir = tempfile.mkdtemp()
    
    try:
        # Process cbioportal to FACETS format
        facets_data = process_cbioportal_to_facets(tmp_seg_path)
        
        # Save FACETS data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as tmp_facets_file:
            facets_data.to_csv(tmp_facets_file.name, sep='\t', index=False)
            tmp_facets_path = tmp_facets_file.name
        
        # Generate CNV matrix using the saved FACETS file
        # generateCNVMatrix(file_type, input_file, project, output_path, folder=False)
        cnv_matrix = generateCNVMatrix(
            file_type="FACETS",
            input_file=tmp_facets_path,
            project="temp_project",
            output_path=temp_output_dir,
            folder=False
        )
        
        # The matrix comes back with features as rows and samples as columns
        # We need to transpose it so samples are rows and features are columns
        if hasattr(cnv_matrix, 'index') and hasattr(cnv_matrix, 'columns'):
            # Remove the first column if it's the MutationType column
            if 'MutationType' in cnv_matrix.columns:
                cnv_matrix = cnv_matrix.set_index('MutationType')
            
            # Transpose so samples are rows and features are columns
            cnv_matrix = cnv_matrix.T
        
        # Ensure proper format (samples as rows, features as columns)
        if cnv_matrix.shape[1] != 28:
            raise ValueError(f"Expected 28 CNA features, got {cnv_matrix.shape[1]}")
        
        return cnv_matrix
        
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_seg_path):
            os.unlink(tmp_seg_path)
        if 'tmp_facets_path' in locals() and os.path.exists(tmp_facets_path):
            os.unlink(tmp_facets_path)
        # Clean up temporary directory
        if os.path.exists(temp_output_dir):
            import shutil
            shutil.rmtree(temp_output_dir)

def _process_matrix_file(uploaded_file, file_extension: str) -> pd.DataFrame:
    """
    Process a pre-computed CNA matrix file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        file_extension: File extension (.tsv or .csv)
        
    Returns:
        pd.DataFrame: Processed CNA matrix
    """
    separator = '\t' if file_extension == '.tsv' else ','
    
    # Read matrix file
    df = pd.read_csv(uploaded_file, sep=separator, index_col=0)
    
    # Basic validation
    if df.empty:
        raise ValueError("Empty matrix file")
    
    # Check for numeric data
    if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
        raise ValueError("Matrix must contain only numeric data")
    
    # Check for expected number of features (28 for CNA consensus signatures)
    if df.shape[1] != 28:
        st.warning(f"Expected 28 CNA features, got {df.shape[1]}. Proceeding with provided features.")
    
    return df

def load_consensus_signatures() -> pd.DataFrame:
    """
    Load consensus signatures from the reference file.
    
    Returns:
        pd.DataFrame: Consensus signatures with CNA categories as index and signatures as columns
    """
    consensus_file_path = Path(__file__).parent / "consensus_signatures.csv"
    
    if not consensus_file_path.exists():
        raise FileNotFoundError(f"Consensus signatures file not found at {consensus_file_path}")
    
    # Load consensus signatures
    consensus_df = pd.read_csv(consensus_file_path, index_col=0)
    
    return consensus_df

def run_signature_fitting(
    cna_matrix: pd.DataFrame,
    use_bootstrap: bool = False,
    bootstrap_iterations: int = 200,
    method: str = 'nnls'
) -> Dict[str, Any]:
    """
    Run signature fitting analysis on the CNA matrix.
    
    Args:
        cna_matrix: CNA matrix with samples as rows and features as columns
        use_bootstrap: Whether to use bootstrap uncertainty estimation
        bootstrap_iterations: Number of bootstrap iterations
        method: Deconvolution method ('nnls' or 'elastic_net')
        
    Returns:
        Dict containing analysis results
    """
    try:
        # Load consensus signatures
        consensus_signatures = load_consensus_signatures()
        
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Preprocess the data
        processed_data = data_processor.preprocess_cna_data(cna_matrix)
        
        # Choose fitting method
        if use_bootstrap:
            fitter = BootstrappedSignatureFitter(
                consensus_signatures=consensus_signatures,
                n_iterations=bootstrap_iterations,
                method=method
            )
        else:
            fitter = ConsensusSignatureFitter(
                consensus_signatures=consensus_signatures,
                method=method
            )
        
        # Fit signatures
        with st.spinner(f"Fitting signatures using {method} method..."):
            if use_bootstrap:
                # Bootstrap fitter returns more values
                point_estimates, base_metrics, mean_activities, ci_lower, ci_upper = fitter.fit(processed_data)
                activities = point_estimates
                
                # Calculate quality metrics
                r2_scores = base_metrics.get('r2_scores', [])
                reconstruction_errors = base_metrics.get('reconstruction_error', [])
                
                # Format results with confidence intervals
                result_dict = {
                    'activities': activities,
                    'mean_r2': base_metrics.get('mean_r2', 0.0),
                    'mean_error': base_metrics.get('mean_reconstruction_error', 0.0),
                    'method': method,
                    'use_bootstrap': use_bootstrap,
                    'bootstrap_iterations': bootstrap_iterations,
                    'confidence_intervals': {
                        'lower': ci_lower,
                        'upper': ci_upper,
                        'mean': mean_activities
                    }
                }
            else:
                # Regular fitter returns activities and metrics
                activities, metrics = fitter.fit(processed_data)
                
                # Calculate quality metrics
                r2_scores = metrics.get('r2_scores', [])
                reconstruction_errors = metrics.get('reconstruction_error', [])
                
                mean_r2 = metrics.get('mean_r2', 0.0)
                mean_error = metrics.get('mean_reconstruction_error', 0.0)
                
                # Format results
                result_dict = {
                    'activities': activities,
                    'mean_r2': mean_r2,
                    'mean_error': mean_error,
                    'method': method,
                    'use_bootstrap': use_bootstrap
                }
        
        return result_dict
        
    except Exception as e:
        raise RuntimeError(f"Error during signature fitting: {str(e)}")

def validate_cna_matrix(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that a DataFrame is a proper CNA matrix.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Matrix is empty"
    
    # Check for numeric data
    if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
        return False, "Matrix must contain only numeric data"
    
    # Check for missing values
    if df.isnull().any().any():
        return False, "Matrix contains missing values"
    
    # Check sample names (should be strings)
    if not df.index.dtype == 'object':
        return False, "Sample names should be strings"
    
    # Check for duplicate samples
    if df.index.duplicated().any():
        return False, "Duplicate sample names found"
    
    return True, "Matrix is valid"

def get_example_data() -> Dict[str, pd.DataFrame]:
    """
    Generate example data for testing the application.
    
    Returns:
        Dict with example matrices
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic CNA matrix
    n_samples = 20
    
    # Create sample names
    sample_names = [f"Sample_{i+1:03d}" for i in range(n_samples)]
    
    # Create feature names (proper CNA categories)
    feature_names = [
        "0:homdel:0-100kb", "0:homdel:100kb-1Mb", "0:homdel:>1Mb",
        "1:LOH:0-100kb", "1:LOH:100kb-1Mb", "1:LOH:1Mb-10Mb", "1:LOH:10Mb-40Mb", "1:LOH:>40Mb",
        "2:het:0-100kb", "2:het:100kb-1Mb", "2:het:1Mb-10Mb", "2:het:10Mb-40Mb", "2:het:>40Mb",
        "3-4:het:0-100kb", "3-4:het:100kb-1Mb", "3-4:het:1Mb-10Mb", "3-4:het:10Mb-40Mb", "3-4:het:>40Mb",
        "5-8:het:0-100kb", "5-8:het:100kb-1Mb", "5-8:het:1Mb-10Mb", "5-8:het:10Mb-40Mb", "5-8:het:>40Mb",
        "9+:het:0-100kb", "9+:het:100kb-1Mb", "9+:het:1Mb-10Mb", "9+:het:10Mb-40Mb", "9+:het:>40Mb"
    ]
    
    n_features = len(feature_names)
    
    # Generate synthetic data with realistic CNA patterns
    data = np.random.poisson(lam=2.0, size=(n_samples, n_features)).astype(float)
    
    # Add some biological structure
    # Make homozygous deletions rarer
    data[:, 0:3] = np.random.poisson(lam=0.1, size=(n_samples, 3))
    
    # Make heterozygous patterns more common
    data[:, 8:13] = np.random.poisson(lam=5.0, size=(n_samples, 5))
    
    # Add some correlation structure within copy number classes
    for i in range(0, n_features, 5):
        end_idx = min(i + 5, n_features)
        corr_factor = np.random.uniform(0.8, 1.2, size=(n_samples, 1))
        data[:, i:end_idx] *= corr_factor
    
    # Create DataFrame
    example_matrix = pd.DataFrame(
        data,
        index=sample_names,
        columns=feature_names
    )
    
    return {
        'example_matrix': example_matrix
    }