"""
Visualization module for consensus CNA signature fitting results.

Provides comprehensive plotting capabilities for signature activities,
validation metrics, and comparison analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
import logging
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class SignatureVisualizer:
    """
    Creates visualizations for consensus signature fitting results.
    
    Provides methods for plotting signature activities, validation metrics,
    correlations, and comparison analyses.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300,
                 save_format: str = 'png'):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
            save_format: Format for saved figures ('png', 'pdf', 'svg')
        """
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format
        
        # Color schemes
        self.signature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.quality_colors = {'high': '#2ca02c', 'medium': '#ff7f0e', 'low': '#d62728'}
        
    def plot_signature_activities(self,
                                activities: pd.DataFrame,
                                output_path: Union[str, Path] = None,
                                title: str = "Consensus Signature Activities",
                                sample_labels: bool = False,
                                sort_by: str = None) -> plt.Figure:
        """
        Plot signature activities as a heatmap.
        
        Args:
            activities: DataFrame with samples as rows, signatures as columns
            output_path: Path to save the plot
            title: Plot title
            sample_labels: Whether to show sample labels on y-axis
            sort_by: Signature to sort samples by
            
        Returns:
            figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Prepare data
        plot_data = activities.copy()
        
        # Sort samples if requested
        if sort_by and sort_by in plot_data.columns:
            plot_data = plot_data.sort_values(by=sort_by, ascending=False)
        
        # Create heatmap
        sns.heatmap(
            plot_data.T,
            ax=ax,
            cmap='viridis',
            cbar_kws={'label': 'Activity Level'},
            xticklabels=sample_labels,
            yticklabels=True
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Samples', fontsize=12)
        ax.set_ylabel('Consensus Signatures', fontsize=12)
        
        # Rotate y-axis labels for better readability
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        if sample_labels:
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def plot_signature_distribution(self,
                                  activities: pd.DataFrame,
                                  output_path: Union[str, Path] = None,
                                  title: str = "Signature Activity Distributions") -> plt.Figure:
        """
        Plot distribution of signature activities using box plots.
        
        Args:
            activities: DataFrame with signature activities
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            figure: Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2), dpi=self.dpi)
        
        # Box plot
        activities.boxplot(ax=axes[0], patch_artist=True)
        axes[0].set_title(f"{title} - Box Plot", fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Activity Level', fontsize=10)
        axes[0].set_xlabel('Consensus Signatures', fontsize=10)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Violin plot
        activities_melted = activities.melt(var_name='Signature', value_name='Activity')
        sns.violinplot(data=activities_melted, x='Signature', y='Activity', ax=axes[1])
        axes[1].set_title(f"{title} - Violin Plot", fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Activity Level', fontsize=10)
        axes[1].set_xlabel('Consensus Signatures', fontsize=10)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def plot_correlation_matrix(self,
                              activities: pd.DataFrame,
                              reference_activities: pd.DataFrame = None,
                              output_path: Union[str, Path] = None,
                              title: str = "Signature Activity Correlations") -> plt.Figure:
        """
        Plot correlation matrix between fitted and reference activities.
        
        Args:
            activities: Fitted signature activities
            reference_activities: Reference activities for comparison
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            figure: Matplotlib figure object
        """
        if reference_activities is not None:
            # Align samples
            common_samples = activities.index.intersection(reference_activities.index)
            
            if len(common_samples) == 0:
                raise ValueError("No common samples found between fitted and reference activities")
            
            activities_aligned = activities.loc[common_samples]
            reference_aligned = reference_activities.loc[common_samples]
            
            # Compute correlations between corresponding signatures
            correlations = []
            for sig in activities.columns:
                if sig in reference_aligned.columns:
                    corr = np.corrcoef(activities_aligned[sig], reference_aligned[sig])[0, 1]
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
            
            # Create correlation matrix plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]), dpi=self.dpi)
            
            # Correlation bar plot
            signatures = activities.columns
            bars = ax1.bar(range(len(signatures)), correlations, color=self.signature_colors[:len(signatures)])
            ax1.set_title("Signature Correlations\n(Fitted vs Reference)", fontsize=12, fontweight='bold')
            ax1.set_ylabel('Pearson Correlation', fontsize=10)
            ax1.set_xlabel('Consensus Signatures', fontsize=10)
            ax1.set_xticks(range(len(signatures)))
            ax1.set_xticklabels(signatures, rotation=45)
            ax1.set_ylim([0, 1])
            ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good correlation (>0.8)')
            ax1.legend()
            
            # Add correlation values on bars
            for bar, corr in zip(bars, correlations):
                if not np.isnan(corr):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Scatter plot for best correlation
            best_idx = np.nanargmax(correlations)
            best_sig = signatures[best_idx]
            
            ax2.scatter(reference_aligned[best_sig], activities_aligned[best_sig], 
                       alpha=0.6, s=30)
            ax2.plot([reference_aligned[best_sig].min(), reference_aligned[best_sig].max()],
                    [reference_aligned[best_sig].min(), reference_aligned[best_sig].max()],
                    'r--', alpha=0.8)
            ax2.set_xlabel(f'Reference {best_sig}', fontsize=10)
            ax2.set_ylabel(f'Fitted {best_sig}', fontsize=10)
            ax2.set_title(f'Best Correlation: {best_sig}\n(r = {correlations[best_idx]:.3f})', 
                         fontsize=12, fontweight='bold')
            
        else:
            # Plot correlation within fitted activities
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            corr_matrix = activities.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def plot_quality_metrics(self,
                           metrics: Dict,
                           output_path: Union[str, Path] = None,
                           title: str = "Fitting Quality Metrics") -> plt.Figure:
        """
        Plot quality metrics for signature fitting.
        
        Args:
            metrics: Dictionary containing quality metrics
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            figure: Matplotlib figure object
        """
        # Set font to Arial and configure sizes
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=self.dpi)
        
        # R² distribution
        r2_scores = metrics.get('r2_scores', [])
        if len(r2_scores) > 0:
            axes[0, 0].hist(r2_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(np.mean(r2_scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(r2_scores):.3f}')
            axes[0, 0].set_title('R² Score Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('R² Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # Reconstruction error distribution
        recon_errors = metrics.get('reconstruction_error', [])
        if len(recon_errors) > 0:
            axes[0, 1].hist(recon_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].axvline(np.mean(recon_errors), color='red', linestyle='--',
                              label=f'Mean: {np.mean(recon_errors):.3f}')
            axes[0, 1].set_title('Reconstruction Error Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Mean Squared Error')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # Quality categories
        if len(r2_scores) > 0:
            quality_categories = ['High (R²>0.8)', 'Medium (0.5<R²≤0.8)', 'Low (R²≤0.5)']
            high_quality = np.sum(np.array(r2_scores) > 0.8)
            medium_quality = np.sum((np.array(r2_scores) > 0.5) & (np.array(r2_scores) <= 0.8))
            low_quality = np.sum(np.array(r2_scores) <= 0.5)
            
            quality_counts = [high_quality, medium_quality, low_quality]
            colors = [self.quality_colors['high'], self.quality_colors['medium'], self.quality_colors['low']]
            
            wedges, texts, autotexts = axes[1, 0].pie(quality_counts, labels=quality_categories, 
                                                     colors=colors, autopct='%1.1f%%', startangle=90,
                                                     pctdistance=0.85)
            # Adjust font size and positioning for percentage labels
            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_color('white')
                autotext.set_weight('bold')
            axes[1, 0].set_title('Fitting Quality Distribution', fontweight='bold')
        
        # Summary statistics
        if len(r2_scores) > 0 and len(recon_errors) > 0:
            stats_text = f"""Summary Statistics:

R² Score:
  Mean: {np.mean(r2_scores):.3f}
  Median: {np.median(r2_scores):.3f}
  Std: {np.std(r2_scores):.3f}

Reconstruction Error:
  Mean: {np.mean(recon_errors):.3f}
  Median: {np.median(recon_errors):.3f}
  Std: {np.std(recon_errors):.3f}

Sample Count: {len(r2_scores)}"""
            
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 1].set_title('Summary Statistics', fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    def plot_signature_contributions_stacked(self,
                                           activities: pd.DataFrame,
                                           output_path: Union[str, Path] = None,
                                           title: str = "Signature Contributions per sample",
                                           max_samples_per_plot: int = 100,
                                           sort_by_total: bool = True) -> plt.Figure:
        """
        Plot stacked bar plot showing relative signature contributions per sample.
        Creates multi-page PDF with ALL samples, showing 100 samples per page.
        
        Args:
            activities: Signature activities DataFrame (samples x signatures)
            output_path: Path to save the plot (will be saved as PDF)
            title: Plot title
            max_samples_per_plot: Maximum number of samples per plot page
            sort_by_total: Whether to sort samples by total activity
            
        Returns:
            figure: Last matplotlib figure object (for compatibility)
        """
        # Set font to Arial
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['font.size'] = 12
        
        # Calculate relative contributions (normalize each sample to 1.0)
        relative_activities = activities.div(activities.sum(axis=1), axis=0)
        
        # Sort samples by total activity if requested
        if sort_by_total:
            total_activity = activities.sum(axis=1)
            relative_activities = relative_activities.loc[total_activity.sort_values(ascending=False).index]
        
        # Calculate number of plots needed
        n_samples = len(relative_activities)
        n_plots = int(np.ceil(n_samples / max_samples_per_plot))
        
        print(f"   Creating {n_plots} plots for {n_samples} samples ({max_samples_per_plot} samples per plot)")
        
        # Determine output path - force PDF format
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix.lower() != '.pdf':
                output_path = output_path.with_suffix('.pdf')
        
        # Create multi-page PDF
        last_fig = None
        
        with PdfPages(output_path) as pdf:
            for plot_idx in range(n_plots):
                # Get samples for this plot
                start_idx = plot_idx * max_samples_per_plot
                end_idx = min(start_idx + max_samples_per_plot, n_samples)
                plot_data = relative_activities.iloc[start_idx:end_idx]
                
                # Create figure with minimal white space
                fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
                
                # Create stacked bar plot
                bottom = np.zeros(len(plot_data))
                
                # Use Set3 colormap for better distinction
                n_sigs = len(plot_data.columns)
                colors = plt.cm.Set3(np.linspace(0, 1, n_sigs))
                
                # Rename signature columns to use CON format
                renamed_columns = []
                for signature in plot_data.columns:
                    if 'consensus_' in signature.lower():
                        new_name = signature.replace('consensus_', 'CON').replace('Consensus_', 'CON')
                    else:
                        new_name = signature
                    renamed_columns.append(new_name)
                
                for i, (signature, renamed_sig) in enumerate(zip(plot_data.columns, renamed_columns)):
                    values = plot_data[signature].values
                    bars = ax.bar(range(len(plot_data)), values, bottom=bottom, 
                                 label=renamed_sig, color=colors[i], alpha=0.9, width=0.95)
                    bottom += values
                
                # Customize plot with larger fonts and minimal spacing
                ax.set_xlabel('Samples', fontsize=16, fontweight='bold', fontfamily='Arial')
                ax.set_ylabel('Relative Signature Contribution', fontsize=20, fontweight='bold', fontfamily='Arial')
                
                # Create page-specific title
                page_title = f"{title}, samples {start_idx + 1}-{end_idx}"
                ax.set_title(page_title, fontsize=20, fontweight='bold', fontfamily='Arial', pad=40)
                
                # Set x-axis labels with intelligent spacing
                sample_labels = plot_data.index
                n_samples_in_plot = len(sample_labels)
                
                if n_samples_in_plot <= 20:
                    # Show all labels for small datasets
                    ax.set_xticks(range(len(sample_labels)))
                    ax.set_xticklabels(sample_labels, rotation=45, ha='right', 
                                      fontsize=10, fontfamily='Arial')
                elif n_samples_in_plot <= 50:
                    # Show every 2nd label
                    step = 2
                    tick_positions = range(0, len(sample_labels), step)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([sample_labels[i] for i in tick_positions], 
                                      rotation=45, ha='right', fontsize=9, fontfamily='Arial')
                else:
                    # Show every 10th label for larger datasets
                    step = 10
                    tick_positions = range(0, len(sample_labels), step)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([sample_labels[i] for i in tick_positions], 
                                      rotation=45, ha='right', fontsize=8, fontfamily='Arial')
                # Set y-axis to 0-1 with larger font
                ax.set_ylim(0, 1)
                ax.tick_params(axis='y', labelsize=14)
                ax.tick_params(axis='x', labelsize=12)
                
                # Remove x-axis padding by setting exact limits
                ax.set_xlim(-0.5, len(plot_data) - 0.5)
                # Add legend with larger font below the title
                if plot_idx > -1:  # Only add legend to first page
                    legend = ax.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', 
                                      fontsize=16, ncol=min(len(plot_data.columns), 5))
                    # legend.set_title('Signatures', prop={'size': 16, 'weight': 'bold'})
                    # Set font for legend text
                    for text in legend.get_texts():
                        text.set_fontfamily('Arial')
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
                
                # Minimize white space
                plt.subplots_adjust(left=0.08, right=0.85, top=0.85, bottom=0.15, 
                                   hspace=0.1, wspace=0.1)
                
                # # Add summary statistics as text box
                # mean_activities = activities.iloc[start_idx:end_idx].mean(axis=1)
                # stats_text = f"Samples: {end_idx - start_idx}\nMean total activity: {mean_activities.mean():.1f} ± {mean_activities.std():.1f}"
                # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                #        verticalalignment='top', fontsize=11, fontfamily='Arial',
                #        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
                
                # Save page to PDF
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                last_fig = fig
                # Close figure to save memory (except the last one for return)
                if plot_idx < n_plots - 1:
                    plt.close(fig)
        
        print(f"   ✓ Multi-page PDF created: {output_path}")
        print(f"   ✓ Total pages: {n_plots}")
        print(f"   ✓ Total samples plotted: {n_samples}")
        
        # Reset matplotlib settings
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['font.size'] = 10
        
        return last_fig
    
    def plot_method_comparison(self,
                             comparison_results: pd.DataFrame,
                             output_path: Union[str, Path] = None,
                             title: str = "Method Comparison") -> plt.Figure:
        """
        Plot comparison between different fitting methods.
        
        Args:
            comparison_results: DataFrame with method comparison results
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            figure: Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1]), dpi=self.dpi)
        
        # R² comparison
        if 'mean_r2' in comparison_results.columns:
            bars1 = axes[0].bar(comparison_results['method'], comparison_results['mean_r2'],
                               color=self.signature_colors[:len(comparison_results)])
            axes[0].set_title('Mean R² Score by Method', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Mean R² Score')
            axes[0].set_xlabel('Method')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars1, comparison_results['mean_r2']):
                if not np.isnan(value):
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')
        
        # Reconstruction error comparison
        if 'mean_reconstruction_error' in comparison_results.columns:
            bars2 = axes[1].bar(comparison_results['method'], comparison_results['mean_reconstruction_error'],
                               color=self.signature_colors[:len(comparison_results)])
            axes[1].set_title('Mean Reconstruction Error by Method', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Mean Reconstruction Error')
            axes[1].set_xlabel('Method')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars2, comparison_results['mean_reconstruction_error']):
                if not np.isnan(value):
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                                f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def create_summary_report(self,
                            activities: pd.DataFrame,
                            metrics: Dict,
                            reference_activities: pd.DataFrame = None,
                            output_dir: Union[str, Path] = None,
                            sample_prefix: str = "validation") -> List[plt.Figure]:
        """
        Create a comprehensive summary report with multiple plots.
        
        Args:
            activities: Fitted signature activities
            metrics: Quality metrics dictionary
            reference_activities: Reference activities for comparison
            output_dir: Directory to save plots
            sample_prefix: Prefix for output filenames
            
        Returns:
            figures: List of created figure objects
        """
        figures = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Signature activities heatmap
        fig1 = self.plot_signature_activities(
            activities,
            output_path=output_dir / f"{sample_prefix}_activities_heatmap.{self.save_format}" if output_dir else None,
            title="Consensus Signature Activities"
        )
        figures.append(fig1)
        
        # 2. Activity distributions
        fig2 = self.plot_signature_distribution(
            activities,
            output_path=output_dir / f"{sample_prefix}_activity_distributions.{self.save_format}" if output_dir else None,
            title="Signature Activity Distributions"
        )
        figures.append(fig2)
        
        # 3. Quality metrics
        fig3 = self.plot_quality_metrics(
            metrics,
            output_path=output_dir / f"{sample_prefix}_quality_metrics.{self.save_format}" if output_dir else None,
            title="Fitting Quality Metrics"
        )
        figures.append(fig3)
        
        # 4. Correlations (if reference provided)
        if reference_activities is not None:
            fig4 = self.plot_correlation_matrix(
                activities,
                reference_activities,
                output_path=output_dir / f"{sample_prefix}_correlations.{self.save_format}" if output_dir else None,
                title="Fitted vs Reference Correlations"
            )
            figures.append(fig4)
        
        return figures
    
    def _save_figure(self, fig: plt.Figure, output_path: Union[str, Path]) -> None:
        """Save figure to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        logger.info(f"Saved plot to {output_path}")
    
    def close_all_figures(self) -> None:
        """Close all open matplotlib figures to free memory."""
        plt.close('all') 