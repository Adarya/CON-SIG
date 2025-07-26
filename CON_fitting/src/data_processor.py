"""
Data processing and validation module for CNA signature fitting.

Handles input validation, format conversion, and preprocessing of CNA data
to ensure compatibility with the consensus signature fitting framework.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Union, Tuple, Optional
import warnings

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, validation, and preprocessing for CNA signature fitting.
    
    Supports multiple input formats and ensures data quality and compatibility
    with the consensus signature framework.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data processor.
        
        Args:
            verbose: Whether to print processing information
        """
        self.verbose = verbose
        self.supported_formats = ['.csv', '.tsv', '.txt', '.xlsx']
        
    def load_cna_data(self, 
                      file_path: Union[str, Path],
                      sample_col: str = None,
                      transpose: bool = False) -> pd.DataFrame:
        """
        Load CNA data from file.
        
        Args:
            file_path: Path to the data file
            sample_col: Column name containing sample IDs (if None, uses index)
            transpose: Whether to transpose the data after loading
            
        Returns:
            cna_data: DataFrame with samples as rows, CNA categories as columns
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        if self.verbose:
            logger.info(f"Loading CNA data from {file_path}")
        
        # Load data based on file format
        if file_path.suffix == '.csv':
            data = pd.read_csv(file_path, index_col=0)
        elif file_path.suffix in ['.tsv', '.txt']:
            data = pd.read_csv(file_path, sep='\t', index_col=0)
        elif file_path.suffix == '.xlsx':
            data = pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        # Handle sample column if specified
        if sample_col is not None:
            if sample_col in data.columns:
                data = data.set_index(sample_col)
            else:
                raise ValueError(f"Sample column '{sample_col}' not found in data")
        
        # Transpose if requested
        if transpose:
            data = data.T
        
        if self.verbose:
            logger.info(f"Loaded data with shape {data.shape} "
                       f"({len(data)} samples, {len(data.columns)} categories)")
        
        return data
    
    def load_consensus_signatures(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load consensus signatures from file.
        
        Args:
            file_path: Path to consensus signatures file
            
        Returns:
            signatures: DataFrame with CNA categories as rows, signatures as columns
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Consensus signatures file not found: {file_path}")
        
        if self.verbose:
            logger.info(f"Loading consensus signatures from {file_path}")
        
        signatures = pd.read_csv(file_path, index_col=0)
        
        if self.verbose:
            logger.info(f"Loaded {signatures.shape[1]} consensus signatures "
                       f"with {signatures.shape[0]} CNA categories")
        
        return signatures
    
    def validate_cna_data(self, cna_data: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate CNA data format and content.
        
        Args:
            cna_data: CNA data to validate
            
        Returns:
            is_valid: Whether the data passes validation
            issues: List of validation issues found
        """
        issues = []
        
        # Check data shape
        if cna_data.empty:
            issues.append("Data is empty")
            return False, issues
        
        if cna_data.shape[0] == 0:
            issues.append("No samples found")
        
        if cna_data.shape[1] == 0:
            issues.append("No CNA categories found")
        
        # Check for missing values
        missing_pct = (cna_data.isnull().sum().sum() / cna_data.size) * 100
        if missing_pct > 50:
            issues.append(f"High percentage of missing values: {missing_pct:.1f}%")
        elif missing_pct > 10:
            issues.append(f"Moderate percentage of missing values: {missing_pct:.1f}%")
        
        # Check data types
        non_numeric_cols = cna_data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            issues.append(f"Non-numeric columns found: {list(non_numeric_cols)}")
        
        # Check for negative values
        if (cna_data < 0).any().any():
            negative_count = (cna_data < 0).sum().sum()
            issues.append(f"Negative values found: {negative_count} entries")
        
        # Check for extreme values
        if cna_data.select_dtypes(include=[np.number]).max().max() > 1000:
            issues.append("Very large values detected (>1000)")
        
        # Check index and column names
        if cna_data.index.duplicated().any():
            dup_count = cna_data.index.duplicated().sum()
            issues.append(f"Duplicate sample names: {dup_count}")
        
        if cna_data.columns.duplicated().any():
            dup_count = cna_data.columns.duplicated().sum()
            issues.append(f"Duplicate CNA category names: {dup_count}")
        
        is_valid = len(issues) == 0
        
        if self.verbose:
            if is_valid:
                logger.info("Data validation passed")
            else:
                logger.warning(f"Data validation found {len(issues)} issues")
                for issue in issues:
                    logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def preprocess_cna_data(self, 
                           cna_data: pd.DataFrame,
                           reference_categories: pd.Index = None,
                           fill_missing: bool = True,
                           clip_negative: bool = True,
                           remove_zero_samples: bool = False,
                           min_signal_threshold: float = 0.0) -> pd.DataFrame:
        """
        Preprocess CNA data for signature fitting.
        
        Args:
            cna_data: Input CNA data
            reference_categories: Reference CNA categories to align with
            fill_missing: Whether to fill missing values
            clip_negative: Whether to clip negative values to zero
            remove_zero_samples: Whether to remove samples with no signal
            min_signal_threshold: Minimum total signal required per sample
            
        Returns:
            processed_data: Preprocessed CNA data
        """
        if self.verbose:
            logger.info("Preprocessing CNA data")
        
        processed_data = cna_data.copy()
        
        # Remove duplicate samples and categories
        if processed_data.index.duplicated().any():
            logger.warning("Removing duplicate samples")
            processed_data = processed_data[~processed_data.index.duplicated()]
        
        if processed_data.columns.duplicated().any():
            logger.warning("Removing duplicate CNA categories")
            processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]
        
        # Handle missing values
        if fill_missing and processed_data.isnull().any().any():
            if self.verbose:
                missing_count = processed_data.isnull().sum().sum()
                logger.info(f"Filling {missing_count} missing values with zeros")
            processed_data = processed_data.fillna(0)
        
        # Handle negative values
        if clip_negative and (processed_data < 0).any().any():
            if self.verbose:
                negative_count = (processed_data < 0).sum().sum()
                logger.info(f"Clipping {negative_count} negative values to zero")
            processed_data = processed_data.clip(lower=0)
        
        # Align with reference categories if provided
        if reference_categories is not None:
            missing_categories = set(reference_categories) - set(processed_data.columns)
            extra_categories = set(processed_data.columns) - set(reference_categories)
            
            if missing_categories:
                if self.verbose:
                    logger.info(f"Adding {len(missing_categories)} missing categories with zeros")
                for cat in missing_categories:
                    processed_data[cat] = 0.0
            
            if extra_categories:
                if self.verbose:
                    logger.info(f"Removing {len(extra_categories)} extra categories")
            
            # Reorder columns to match reference
            processed_data = processed_data.reindex(columns=reference_categories, fill_value=0)
        
        # Remove samples with insufficient signal
        if remove_zero_samples or min_signal_threshold > 0:
            sample_totals = processed_data.sum(axis=1)
            low_signal_mask = sample_totals <= min_signal_threshold
            
            if low_signal_mask.any():
                n_removed = low_signal_mask.sum()
                if self.verbose:
                    logger.info(f"Removing {n_removed} samples with signal <= {min_signal_threshold}")
                processed_data = processed_data[~low_signal_mask]
        
        if self.verbose:
            logger.info(f"Preprocessing completed. Final shape: {processed_data.shape}")
        
        return processed_data
    
    def create_random_subset(self, 
                           cna_data: pd.DataFrame,
                           n_samples: int,
                           random_state: int = 42) -> pd.DataFrame:
        """
        Create a random subset of samples for testing.
        
        Args:
            cna_data: Input CNA data
            n_samples: Number of samples to select
            random_state: Random seed for reproducibility
            
        Returns:
            subset_data: Random subset of the data
        """
        np.random.seed(random_state)
        
        if n_samples >= len(cna_data):
            logger.warning(f"Requested {n_samples} samples but only {len(cna_data)} available")
            return cna_data.copy()
        
        selected_indices = np.random.choice(cna_data.index, size=n_samples, replace=False)
        subset_data = cna_data.loc[selected_indices]
        
        if self.verbose:
            logger.info(f"Created random subset with {len(subset_data)} samples")
        
        return subset_data
    
    def save_processed_data(self, 
                          data: pd.DataFrame,
                          output_path: Union[str, Path],
                          format: str = 'csv') -> None:
        """
        Save processed data to file.
        
        Args:
            data: Data to save
            output_path: Output file path
            format: Output format ('csv', 'tsv', 'xlsx')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            data.to_csv(output_path)
        elif format == 'tsv':
            data.to_csv(output_path, sep='\t')
        elif format == 'xlsx':
            data.to_excel(output_path)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        if self.verbose:
            logger.info(f"Saved data to {output_path}")
    
    def get_data_summary(self, cna_data: pd.DataFrame) -> dict:
        """
        Generate summary statistics for CNA data.
        
        Args:
            cna_data: CNA data to summarize
            
        Returns:
            summary: Dictionary with summary statistics
        """
        summary = {
            'n_samples': len(cna_data),
            'n_categories': len(cna_data.columns),
            'total_signal': cna_data.sum().sum(),
            'mean_signal_per_sample': cna_data.sum(axis=1).mean(),
            'median_signal_per_sample': cna_data.sum(axis=1).median(),
            'missing_values': cna_data.isnull().sum().sum(),
            'negative_values': (cna_data < 0).sum().sum(),
            'zero_samples': (cna_data.sum(axis=1) == 0).sum(),
            'min_value': cna_data.min().min(),
            'max_value': cna_data.max().max()
        }
        
        return summary 