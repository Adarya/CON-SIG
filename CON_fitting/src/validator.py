"""
Validation module for consensus CNA signature fitting.

Provides comprehensive validation methods to compare fitted activities
with reference data and assess fitting quality.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class SignatureValidator:
    """
    Validates consensus signature fitting results against reference data.
    
    Provides comprehensive metrics for assessing the quality and accuracy
    of signature activity fitting.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the validator.
        
        Args:
            verbose: Whether to print validation information
        """
        self.verbose = verbose
        
    def validate_activities(self,
                          fitted_activities: pd.DataFrame,
                          reference_activities: pd.DataFrame) -> Dict:
        """
        Comprehensive validation of fitted activities against reference.
        
        Args:
            fitted_activities: Fitted signature activities
            reference_activities: Reference (ground truth) activities
            
        Returns:
            validation_results: Dictionary containing validation metrics
        """
        if self.verbose:
            logger.info("Starting comprehensive activity validation")
        
        # Align data
        fitted_aligned, reference_aligned = self._align_activities(
            fitted_activities, reference_activities
        )
        
        # Compute validation metrics
        results = {
            'sample_metrics': self._compute_sample_metrics(fitted_aligned, reference_aligned),
            'signature_metrics': self._compute_signature_metrics(fitted_aligned, reference_aligned),
            'global_metrics': self._compute_global_metrics(fitted_aligned, reference_aligned),
            'distribution_metrics': self._compute_distribution_metrics(fitted_aligned, reference_aligned),
            'rank_metrics': self._compute_rank_metrics(fitted_aligned, reference_aligned)
        }
        
        # Add summary
        results['summary'] = self._create_validation_summary(results)
        
        if self.verbose:
            self._print_validation_summary(results['summary'])
        
        return results
    
    def _align_activities(self,
                         fitted: pd.DataFrame,
                         reference: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align fitted and reference activities for comparison."""
        # Find common samples and signatures
        common_samples = fitted.index.intersection(reference.index)
        common_signatures = fitted.columns.intersection(reference.columns)
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between fitted and reference activities")
        
        if len(common_signatures) == 0:
            raise ValueError("No common signatures found between fitted and reference activities")
        
        if self.verbose:
            logger.info(f"Aligned data: {len(common_samples)} samples, {len(common_signatures)} signatures")
            if len(common_samples) < len(fitted):
                logger.warning(f"Missing {len(fitted) - len(common_samples)} samples in reference")
            if len(common_signatures) < len(fitted.columns):
                logger.warning(f"Missing {len(fitted.columns) - len(common_signatures)} signatures in reference")
        
        # Align data
        fitted_aligned = fitted.loc[common_samples, common_signatures]
        reference_aligned = reference.loc[common_samples, common_signatures]
        
        return fitted_aligned, reference_aligned
    
    def _compute_sample_metrics(self,
                              fitted: pd.DataFrame,
                              reference: pd.DataFrame) -> Dict:
        """Compute per-sample validation metrics."""
        n_samples = len(fitted)
        sample_metrics = {
            'sample_correlations': [],
            'sample_mse': [],
            'sample_mae': [],
            'sample_r2': [],
            'sample_names': fitted.index.tolist()
        }
        
        for sample in fitted.index:
            fitted_values = fitted.loc[sample].values
            reference_values = reference.loc[sample].values
            
            # Correlation
            if np.std(fitted_values) > 0 and np.std(reference_values) > 0:
                corr = pearsonr(fitted_values, reference_values)[0]
                sample_metrics['sample_correlations'].append(corr)
            else:
                sample_metrics['sample_correlations'].append(np.nan)
            
            # Error metrics
            sample_metrics['sample_mse'].append(mean_squared_error(reference_values, fitted_values))
            sample_metrics['sample_mae'].append(mean_absolute_error(reference_values, fitted_values))
            
            # R² score
            if np.var(reference_values) > 0:
                r2 = r2_score(reference_values, fitted_values)
                sample_metrics['sample_r2'].append(r2)
            else:
                sample_metrics['sample_r2'].append(np.nan)
        
        return sample_metrics
    
    def _compute_signature_metrics(self,
                                 fitted: pd.DataFrame,
                                 reference: pd.DataFrame) -> Dict:
        """Compute per-signature validation metrics."""
        signature_metrics = {
            'signature_correlations': [],
            'signature_mse': [],
            'signature_mae': [],
            'signature_r2': [],
            'signature_names': fitted.columns.tolist()
        }
        
        for signature in fitted.columns:
            fitted_values = fitted[signature].values
            reference_values = reference[signature].values
            
            # Correlation
            if np.std(fitted_values) > 0 and np.std(reference_values) > 0:
                corr = pearsonr(fitted_values, reference_values)[0]
                signature_metrics['signature_correlations'].append(corr)
            else:
                signature_metrics['signature_correlations'].append(np.nan)
            
            # Error metrics
            signature_metrics['signature_mse'].append(mean_squared_error(reference_values, fitted_values))
            signature_metrics['signature_mae'].append(mean_absolute_error(reference_values, fitted_values))
            
            # R² score
            if np.var(reference_values) > 0:
                r2 = r2_score(reference_values, fitted_values)
                signature_metrics['signature_r2'].append(r2)
            else:
                signature_metrics['signature_r2'].append(np.nan)
        
        return signature_metrics
    
    def _compute_global_metrics(self,
                              fitted: pd.DataFrame,
                              reference: pd.DataFrame) -> Dict:
        """Compute global validation metrics."""
        fitted_flat = fitted.values.flatten()
        reference_flat = reference.values.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(fitted_flat) | np.isnan(reference_flat))
        fitted_valid = fitted_flat[valid_mask]
        reference_valid = reference_flat[valid_mask]
        
        global_metrics = {}
        
        if len(fitted_valid) > 0:
            # Correlation metrics
            global_metrics['global_pearson'] = pearsonr(fitted_valid, reference_valid)[0]
            global_metrics['global_spearman'] = spearmanr(fitted_valid, reference_valid)[0]
            
            # Error metrics
            global_metrics['global_mse'] = mean_squared_error(reference_valid, fitted_valid)
            global_metrics['global_mae'] = mean_absolute_error(reference_valid, fitted_valid)
            global_metrics['global_rmse'] = np.sqrt(global_metrics['global_mse'])
            
            # R² score
            global_metrics['global_r2'] = r2_score(reference_valid, fitted_valid)
            
            # Explained variance
            global_metrics['explained_variance'] = 1 - (np.var(reference_valid - fitted_valid) / np.var(reference_valid))
            
            # Relative error metrics
            global_metrics['mape'] = np.mean(np.abs((reference_valid - fitted_valid) / np.maximum(reference_valid, 1e-8))) * 100
            
        return global_metrics
    
    def _compute_distribution_metrics(self,
                                    fitted: pd.DataFrame,
                                    reference: pd.DataFrame) -> Dict:
        """Compute distribution comparison metrics."""
        from scipy.stats import ks_2samp, wasserstein_distance
        
        distribution_metrics = {}
        
        # Overall distributions
        fitted_flat = fitted.values.flatten()
        reference_flat = reference.values.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(fitted_flat) | np.isnan(reference_flat))
        fitted_valid = fitted_flat[valid_mask]
        reference_valid = reference_flat[valid_mask]
        
        if len(fitted_valid) > 0:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = ks_2samp(fitted_valid, reference_valid)
            distribution_metrics['ks_statistic'] = ks_stat
            distribution_metrics['ks_pvalue'] = ks_pvalue
            
            # Wasserstein distance
            distribution_metrics['wasserstein_distance'] = wasserstein_distance(fitted_valid, reference_valid)
            
            # Distribution moments
            distribution_metrics['mean_diff'] = np.mean(fitted_valid) - np.mean(reference_valid)
            distribution_metrics['std_diff'] = np.std(fitted_valid) - np.std(reference_valid)
            distribution_metrics['skew_diff'] = self._safe_skew(fitted_valid) - self._safe_skew(reference_valid)
        
        # Per-signature distribution metrics
        signature_ks_stats = []
        signature_ks_pvalues = []
        
        for signature in fitted.columns:
            fitted_sig = fitted[signature].dropna().values
            reference_sig = reference[signature].dropna().values
            
            if len(fitted_sig) > 0 and len(reference_sig) > 0:
                ks_stat, ks_pvalue = ks_2samp(fitted_sig, reference_sig)
                signature_ks_stats.append(ks_stat)
                signature_ks_pvalues.append(ks_pvalue)
            else:
                signature_ks_stats.append(np.nan)
                signature_ks_pvalues.append(np.nan)
        
        distribution_metrics['signature_ks_stats'] = signature_ks_stats
        distribution_metrics['signature_ks_pvalues'] = signature_ks_pvalues
        
        return distribution_metrics
    
    def _compute_rank_metrics(self,
                            fitted: pd.DataFrame,
                            reference: pd.DataFrame) -> Dict:
        """Compute rank-based validation metrics."""
        rank_metrics = {}
        
        # Sample ranking correlations
        sample_rank_correlations = []
        for sample in fitted.index:
            fitted_ranks = fitted.loc[sample].rank()
            reference_ranks = reference.loc[sample].rank()
            
            rank_corr = spearmanr(fitted_ranks, reference_ranks)[0]
            sample_rank_correlations.append(rank_corr)
        
        rank_metrics['sample_rank_correlations'] = sample_rank_correlations
        rank_metrics['mean_sample_rank_correlation'] = np.nanmean(sample_rank_correlations)
        
        # Signature ranking correlations
        signature_rank_correlations = []
        for signature in fitted.columns:
            fitted_ranks = fitted[signature].rank()
            reference_ranks = reference[signature].rank()
            
            rank_corr = spearmanr(fitted_ranks, reference_ranks)[0]
            signature_rank_correlations.append(rank_corr)
        
        rank_metrics['signature_rank_correlations'] = signature_rank_correlations
        rank_metrics['mean_signature_rank_correlation'] = np.nanmean(signature_rank_correlations)
        
        # Top-k agreement metrics
        rank_metrics['top_samples_agreement'] = self._compute_top_k_agreement(fitted, reference, k=10, axis=0)
        rank_metrics['top_signatures_agreement'] = self._compute_top_k_agreement(fitted, reference, k=3, axis=1)
        
        return rank_metrics
    
    def _compute_top_k_agreement(self,
                               fitted: pd.DataFrame,
                               reference: pd.DataFrame,
                               k: int = 10,
                               axis: int = 0) -> Dict:
        """Compute top-k agreement between fitted and reference rankings."""
        agreements = []
        
        if axis == 0:  # Top samples for each signature
            for signature in fitted.columns:
                fitted_top_k = fitted[signature].nlargest(k).index
                reference_top_k = reference[signature].nlargest(k).index
                
                intersection = len(set(fitted_top_k) & set(reference_top_k))
                agreement = intersection / k
                agreements.append(agreement)
        
        else:  # Top signatures for each sample
            for sample in fitted.index:
                fitted_top_k = fitted.loc[sample].nlargest(k).index
                reference_top_k = reference.loc[sample].nlargest(k).index
                
                intersection = len(set(fitted_top_k) & set(reference_top_k))
                agreement = intersection / k
                agreements.append(agreement)
        
        return {
            'agreements': agreements,
            'mean_agreement': np.mean(agreements),
            'median_agreement': np.median(agreements)
        }
    
    def _create_validation_summary(self, results: Dict) -> Dict:
        """Create a summary of validation results."""
        summary = {}
        
        # Global performance
        global_metrics = results['global_metrics']
        summary['overall_correlation'] = global_metrics.get('global_pearson', np.nan)
        summary['overall_r2'] = global_metrics.get('global_r2', np.nan)
        summary['overall_rmse'] = global_metrics.get('global_rmse', np.nan)
        
        # Signature performance
        sig_corrs = results['signature_metrics']['signature_correlations']
        summary['mean_signature_correlation'] = np.nanmean(sig_corrs)
        summary['min_signature_correlation'] = np.nanmin(sig_corrs)
        summary['max_signature_correlation'] = np.nanmax(sig_corrs)
        summary['high_quality_signatures'] = np.sum(np.array(sig_corrs) > 0.8)
        
        # Sample performance
        sample_corrs = results['sample_metrics']['sample_correlations']
        summary['mean_sample_correlation'] = np.nanmean(sample_corrs)
        summary['high_quality_samples'] = np.sum(np.array(sample_corrs) > 0.8)
        
        # Distribution similarity
        summary['distribution_similarity'] = 1 - results['distribution_metrics'].get('ks_statistic', 1)
        
        # Rank correlation
        summary['rank_correlation'] = results['rank_metrics'].get('mean_signature_rank_correlation', np.nan)
        
        # Overall quality score (weighted combination)
        weights = {
            'correlation': 0.3,
            'r2': 0.3,
            'rank_correlation': 0.2,
            'distribution_similarity': 0.2
        }
        
        quality_score = 0
        for metric, weight in weights.items():
            if metric == 'correlation':
                value = summary['overall_correlation']
            elif metric == 'r2':
                value = max(0, summary['overall_r2'])  # Ensure non-negative
            elif metric == 'rank_correlation':
                value = summary['rank_correlation']
            elif metric == 'distribution_similarity':
                value = summary['distribution_similarity']
            
            if not np.isnan(value):
                quality_score += weight * value
        
        summary['quality_score'] = quality_score
        
        return summary
    
    def _print_validation_summary(self, summary: Dict) -> None:
        """Print validation summary to console."""
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Overall Correlation:     {summary['overall_correlation']:.3f}")
        print(f"Overall R²:              {summary['overall_r2']:.3f}")
        print(f"Overall RMSE:            {summary['overall_rmse']:.3f}")
        print(f"Quality Score:           {summary['quality_score']:.3f}")
        print("\nSignature Performance:")
        print(f"  Mean Correlation:      {summary['mean_signature_correlation']:.3f}")
        print(f"  Range:                 [{summary['min_signature_correlation']:.3f}, {summary['max_signature_correlation']:.3f}]")
        print(f"  High Quality (r>0.8):  {summary['high_quality_signatures']}")
        print("\nSample Performance:")
        print(f"  Mean Correlation:      {summary['mean_sample_correlation']:.3f}")
        print(f"  High Quality (r>0.8):  {summary['high_quality_samples']}")
        print(f"\nRank Correlation:        {summary['rank_correlation']:.3f}")
        print(f"Distribution Similarity: {summary['distribution_similarity']:.3f}")
        print("="*50)
    
    def save_validation_report(self,
                             validation_results: Dict,
                             output_path: str) -> None:
        """Save detailed validation report to file."""
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("CONSENSUS SIGNATURE VALIDATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Summary
            summary = validation_results['summary']
            f.write("SUMMARY METRICS\n")
            f.write("-"*30 + "\n")
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"{key:30s}: {value:.4f}\n")
                else:
                    f.write(f"{key:30s}: {value}\n")
            f.write("\n")
            
            # Detailed metrics
            f.write("DETAILED METRICS\n")
            f.write("-"*30 + "\n")
            
            # Global metrics
            f.write("\nGlobal Metrics:\n")
            for key, value in validation_results['global_metrics'].items():
                if isinstance(value, float):
                    f.write(f"  {key:25s}: {value:.4f}\n")
            
            # Signature metrics
            f.write("\nPer-Signature Correlations:\n")
            sig_metrics = validation_results['signature_metrics']
            for i, (name, corr) in enumerate(zip(sig_metrics['signature_names'], 
                                               sig_metrics['signature_correlations'])):
                f.write(f"  {name:20s}: {corr:.4f}\n")
        
        if self.verbose:
            logger.info(f"Validation report saved to {output_path}")
    
    def _safe_skew(self, data: np.ndarray) -> float:
        """Compute skewness safely."""
        from scipy.stats import skew
        try:
            return skew(data)
        except:
            return 0.0 