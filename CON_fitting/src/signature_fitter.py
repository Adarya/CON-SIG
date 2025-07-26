"""
Core signature fitting module for consensus CNA signatures.

This module implements multiple deconvolution algorithms to fit consensus signatures
to new CNA data, with emphasis on reproducibility and accuracy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
import warnings
from typing import Tuple, Dict, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusSignatureFitter:
    """
    Fits consensus CNA signatures to new sample data using multiple deconvolution methods.
    
    Supports various algorithms including Non-Negative Least Squares (NNLS),
    Elastic Net regression, and constrained optimization approaches.
    
    Attributes:
        consensus_signatures (pd.DataFrame): Reference consensus signatures
        method (str): Deconvolution method to use
        normalize (bool): Whether to normalize input data
        verbose (bool): Whether to print progress information
    """
    
    def __init__(self, 
                 consensus_signatures: pd.DataFrame,
                 method: str = 'nnls',
                 normalize: bool = True,
                 verbose: bool = True):
        """
        Initialize the signature fitter.
        
        Args:
            consensus_signatures: DataFrame with CNA categories as rows, signatures as columns
            method: Deconvolution method ('nnls', 'elastic_net', 'constrained_ls')
            normalize: Whether to normalize signatures and data
            verbose: Whether to print progress information
        """
        self.consensus_signatures = consensus_signatures.copy()
        self.method = method
        self.normalize = normalize
        self.verbose = verbose
        
        # Validate and prepare signatures
        self._prepare_signatures()
        
        # Store fitted results
        self.fitted_activities_ = None
        self.reconstruction_error_ = None
        self.r2_scores_ = None
        
    def _prepare_signatures(self):
        """Prepare and validate consensus signatures."""
        # Check for missing values
        if self.consensus_signatures.isnull().any().any():
            logger.warning("Missing values found in consensus signatures. Filling with zeros.")
            self.consensus_signatures = self.consensus_signatures.fillna(0)
        
        # Ensure non-negative values
        if (self.consensus_signatures < 0).any().any():
            logger.warning("Negative values found in signatures. Setting to zero.")
            self.consensus_signatures = self.consensus_signatures.clip(lower=0)
        
        # Normalize signatures if requested
        if self.normalize:
            self.consensus_signatures = self.consensus_signatures.div(
                self.consensus_signatures.sum(axis=0), axis=1
            ).fillna(0)
        
        if self.verbose:
            logger.info(f"Loaded {self.consensus_signatures.shape[1]} consensus signatures "
                       f"with {self.consensus_signatures.shape[0]} CNA categories")
    
    def fit(self, 
            cna_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Fit consensus signatures to new CNA data.
        
        Args:
            cna_data: DataFrame with samples as rows, CNA categories as columns
            
        Returns:
            activities: DataFrame with samples as rows, signatures as columns
            metrics: Dictionary containing quality metrics
        """
        if self.verbose:
            logger.info(f"Fitting {len(cna_data)} samples using {self.method} method")
        
        # Validate and prepare input data
        processed_data, sample_names = self._prepare_input_data(cna_data)
        
        # Perform signature fitting
        if self.method == 'nnls':
            activities, metrics = self._fit_nnls(processed_data, sample_names)
        elif self.method == 'elastic_net':
            activities, metrics = self._fit_elastic_net(processed_data, sample_names)
        elif self.method == 'constrained_ls':
            activities, metrics = self._fit_constrained_ls(processed_data, sample_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store results
        self.fitted_activities_ = activities
        self.reconstruction_error_ = metrics['reconstruction_error']
        self.r2_scores_ = metrics['r2_scores']
        
        if self.verbose:
            logger.info(f"Fitting completed. Mean R² = {np.mean(metrics['r2_scores']):.3f}")
        
        return activities, metrics
    
    def _prepare_input_data(self, cna_data: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """Prepare and validate input CNA data."""
        # Store sample names
        sample_names = cna_data.index
        
        # Align CNA categories with consensus signatures
        common_categories = self.consensus_signatures.index.intersection(cna_data.columns)
        
        if len(common_categories) == 0:
            raise ValueError("No common CNA categories found between input data and signatures")
        
        if len(common_categories) < len(self.consensus_signatures.index):
            missing_cats = set(self.consensus_signatures.index) - set(common_categories)
            logger.warning(f"Missing CNA categories in input data: {missing_cats}")
        
        # Align data and signatures
        aligned_data = cna_data.reindex(columns=self.consensus_signatures.index, fill_value=0)
        
        # Convert to numpy array and transpose (samples x categories -> categories x samples)
        data_matrix = aligned_data.values.T
        
        # Handle missing values
        if np.isnan(data_matrix).any():
            logger.warning("Missing values in input data. Filling with zeros.")
            data_matrix = np.nan_to_num(data_matrix, nan=0.0)
        
        # Ensure non-negative values
        if (data_matrix < 0).any():
            logger.warning("Negative values in input data. Setting to zero.")
            data_matrix = np.clip(data_matrix, 0, None)
        
        return data_matrix, sample_names
    
    def _fit_nnls(self, data_matrix: np.ndarray, sample_names: pd.Index) -> Tuple[pd.DataFrame, Dict]:
        """Fit using Non-Negative Least Squares."""
        n_samples = data_matrix.shape[1]
        n_signatures = self.consensus_signatures.shape[1]
        
        activities = np.zeros((n_samples, n_signatures))
        reconstruction_errors = np.zeros(n_samples)
        r2_scores = np.zeros(n_samples)
        
        signatures_matrix = self.consensus_signatures.values
        
        for i in range(n_samples):
            sample_data = data_matrix[:, i]
            
            # Skip if sample has no signal
            if np.sum(sample_data) == 0:
                if self.verbose and i < 5:  # Only warn for first few samples
                    logger.warning(f"Sample {sample_names[i]} has no CNA signal")
                continue
            
            try:
                # Fit NNLS
                activities[i, :], residual = nnls(signatures_matrix, sample_data)
                
                # Compute reconstruction and metrics
                reconstructed = signatures_matrix @ activities[i, :]
                reconstruction_errors[i] = mean_squared_error(sample_data, reconstructed)
                
                # Compute R² (with protection against division by zero)
                ss_tot = np.sum((sample_data - np.mean(sample_data)) ** 2)
                if ss_tot > 0:
                    r2_scores[i] = r2_score(sample_data, reconstructed)
                else:
                    r2_scores[i] = 0.0
                    
            except Exception as e:
                logger.warning(f"Failed to fit sample {sample_names[i]}: {str(e)}")
                continue
        
        # Create results DataFrame
        activities_df = pd.DataFrame(
            activities,
            index=sample_names,
            columns=self.consensus_signatures.columns
        )
        
        metrics = {
            'reconstruction_error': reconstruction_errors,
            'r2_scores': r2_scores,
            'mean_r2': np.mean(r2_scores),
            'mean_reconstruction_error': np.mean(reconstruction_errors)
        }
        
        return activities_df, metrics
    
    def _fit_elastic_net(self, data_matrix: np.ndarray, sample_names: pd.Index) -> Tuple[pd.DataFrame, Dict]:
        """Fit using Elastic Net regression with non-negativity constraints."""
        from sklearn.linear_model import ElasticNet
        
        n_samples = data_matrix.shape[1]
        n_signatures = self.consensus_signatures.shape[1]
        
        activities = np.zeros((n_samples, n_signatures))
        reconstruction_errors = np.zeros(n_samples)
        r2_scores = np.zeros(n_samples)
        
        signatures_matrix = self.consensus_signatures.values
        
        # Initialize Elastic Net with appropriate parameters
        elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.5, positive=True, max_iter=1000)
        
        for i in range(n_samples):
            sample_data = data_matrix[:, i]
            
            if np.sum(sample_data) == 0:
                continue
            
            try:
                # Fit Elastic Net
                elastic_net.fit(signatures_matrix, sample_data)
                activities[i, :] = elastic_net.coef_
                
                # Compute metrics
                reconstructed = signatures_matrix @ activities[i, :]
                reconstruction_errors[i] = mean_squared_error(sample_data, reconstructed)
                
                ss_tot = np.sum((sample_data - np.mean(sample_data)) ** 2)
                if ss_tot > 0:
                    r2_scores[i] = r2_score(sample_data, reconstructed)
                else:
                    r2_scores[i] = 0.0
                    
            except Exception as e:
                logger.warning(f"Failed to fit sample {sample_names[i]} with Elastic Net: {str(e)}")
                continue
        
        activities_df = pd.DataFrame(
            activities,
            index=sample_names,
            columns=self.consensus_signatures.columns
        )
        
        metrics = {
            'reconstruction_error': reconstruction_errors,
            'r2_scores': r2_scores,
            'mean_r2': np.mean(r2_scores),
            'mean_reconstruction_error': np.mean(reconstruction_errors)
        }
        
        return activities_df, metrics
    
    def _fit_constrained_ls(self, data_matrix: np.ndarray, sample_names: pd.Index) -> Tuple[pd.DataFrame, Dict]:
        """Fit using constrained least squares with sum-to-one constraint."""
        from scipy.optimize import minimize
        
        n_samples = data_matrix.shape[1]
        n_signatures = self.consensus_signatures.shape[1]
        
        activities = np.zeros((n_samples, n_signatures))
        reconstruction_errors = np.zeros(n_samples)
        r2_scores = np.zeros(n_samples)
        
        signatures_matrix = self.consensus_signatures.values
        
        def objective(x, signatures, data):
            """Objective function for optimization."""
            return np.sum((signatures @ x - data) ** 2)
        
        # Constraints: non-negativity
        bounds = [(0, None) for _ in range(n_signatures)]
        
        for i in range(n_samples):
            sample_data = data_matrix[:, i]
            
            if np.sum(sample_data) == 0:
                continue
            
            try:
                # Initial guess
                x0 = np.ones(n_signatures) / n_signatures
                
                # Optimize
                result = minimize(
                    objective,
                    x0,
                    args=(signatures_matrix, sample_data),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    activities[i, :] = result.x
                    
                    # Compute metrics
                    reconstructed = signatures_matrix @ activities[i, :]
                    reconstruction_errors[i] = mean_squared_error(sample_data, reconstructed)
                    
                    ss_tot = np.sum((sample_data - np.mean(sample_data)) ** 2)
                    if ss_tot > 0:
                        r2_scores[i] = r2_score(sample_data, reconstructed)
                    else:
                        r2_scores[i] = 0.0
                else:
                    logger.warning(f"Optimization failed for sample {sample_names[i]}")
                    
            except Exception as e:
                logger.warning(f"Failed to fit sample {sample_names[i]} with constrained LS: {str(e)}")
                continue
        
        activities_df = pd.DataFrame(
            activities,
            index=sample_names,
            columns=self.consensus_signatures.columns
        )
        
        metrics = {
            'reconstruction_error': reconstruction_errors,
            'r2_scores': r2_scores,
            'mean_r2': np.mean(r2_scores),
            'mean_reconstruction_error': np.mean(reconstruction_errors)
        }
        
        return activities_df, metrics
    
    def predict(self, activities: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct CNA data from signature activities.
        
        Args:
            activities: DataFrame with samples as rows, signatures as columns
            
        Returns:
            reconstructed_data: DataFrame with samples as rows, CNA categories as columns
        """
        signatures_matrix = self.consensus_signatures.values
        activities_matrix = activities.values
        
        reconstructed = activities_matrix @ signatures_matrix.T
        
        return pd.DataFrame(
            reconstructed,
            index=activities.index,
            columns=self.consensus_signatures.index
        )
    
    def get_signature_contributions(self, sample_name: str) -> pd.Series:
        """
        Get signature contributions for a specific sample.
        
        Args:
            sample_name: Name of the sample
            
        Returns:
            contributions: Series with signature contributions
        """
        if self.fitted_activities_ is None:
            raise ValueError("No fitted activities available. Run fit() first.")
        
        if sample_name not in self.fitted_activities_.index:
            raise ValueError(f"Sample {sample_name} not found in fitted activities")
        
        return self.fitted_activities_.loc[sample_name]
    
    def compare_methods(self, cna_data: pd.DataFrame, 
                       methods: list = None) -> pd.DataFrame:
        """
        Compare different fitting methods on the same data.
        
        Args:
            cna_data: Input CNA data
            methods: List of methods to compare
            
        Returns:
            comparison_results: DataFrame with method comparison metrics
        """
        if methods is None:
            methods = ['nnls', 'elastic_net', 'constrained_ls']
        
        results = []
        original_method = self.method
        
        for method in methods:
            try:
                self.method = method
                activities, metrics = self.fit(cna_data)
                
                results.append({
                    'method': method,
                    'mean_r2': metrics['mean_r2'],
                    'mean_reconstruction_error': metrics['mean_reconstruction_error'],
                    'n_samples': len(cna_data)
                })
                
            except Exception as e:
                logger.warning(f"Method {method} failed: {str(e)}")
                results.append({
                    'method': method,
                    'mean_r2': np.nan,
                    'mean_reconstruction_error': np.nan,
                    'n_samples': len(cna_data)
                })
        
        # Restore original method
        self.method = original_method
        
        return pd.DataFrame(results) 