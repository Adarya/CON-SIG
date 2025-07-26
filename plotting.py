"""
Plotting utilities for the CONSIG web application.

This module provides web-optimized plotting functions that create publication-ready
visualizations of CNA signature analysis results, specifically designed for Streamlit.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
import io
import base64

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing visualization utilities
try:
    from CON_fitting.src.visualizer import SignatureVisualizer
except ImportError:
    SignatureVisualizer = None

# Set up matplotlib for web display
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for Arial font and larger text
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
})

def format_signature_names(sig_names):
    """
    Format signature names to replace consensus_x with CONX format.
    
    Args:
        sig_names: List or array of signature names
        
    Returns:
        List of formatted signature names
    """
    formatted_names = []
    for name in sig_names:
        if isinstance(name, str):
            # Replace consensus_1, consensus_2, etc. with CON1, CON2, etc.
            if name.lower().startswith('consensus_'):
                number = name.split('_')[-1]
                formatted_names.append(f'CON{number}')
            else:
                formatted_names.append(name)
        else:
            formatted_names.append(name)
    return formatted_names

def create_stacked_bar_plot(
    activities_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "CNA Signature Activities",
    max_samples: int = 50
) -> plt.Figure:
    """
    Create a stacked bar plot of signature activities optimized for web display.
    
    Args:
        activities_df: DataFrame with samples as rows and signatures as columns
        figsize: Figure size (width, height)
        title: Plot title
        max_samples: Maximum number of samples to display (for readability)
        
    Returns:
        matplotlib Figure object
    """
    # Limit number of samples for readability
    if len(activities_df) > max_samples:
        activities_df = activities_df.head(max_samples)
    
    # Create figure with no padding
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get signature columns (assuming they start with 'Signature' or 'SBS')
    sig_columns = [col for col in activities_df.columns if 'signature' in col.lower() or 'sbs' in col.lower()]
    
    # If no signature columns found, use all columns
    if not sig_columns:
        sig_columns = activities_df.columns.tolist()
    
    # Create stacked bar plot
    bottom = np.zeros(len(activities_df))
    
    # Generate colors for signatures
    colors = plt.cm.Set3(np.linspace(0, 1, len(sig_columns)))
    
    # Format signature names for display
    formatted_sig_names = format_signature_names(sig_columns)
    
    bars = []
    for i, (sig, formatted_name) in enumerate(zip(sig_columns, formatted_sig_names)):
        if sig in activities_df.columns:
            values = activities_df[sig].values
            bar = ax.bar(
                range(len(activities_df)), 
                values, 
                bottom=bottom, 
                label=formatted_name,
                color=colors[i],
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5,
                width=1.0
            )
            bars.append(bar)
            bottom += values
    
    # Customize plot
    ax.set_xlabel('Samples', fontsize=18, fontweight='bold')
    ax.set_ylabel('Signature Activity', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Set x-axis labels with no gaps and smaller font size
    sample_names = activities_df.index.tolist()
    ax.set_xticks(range(len(sample_names)))
    ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=12)
    ax.set_xlim(-0.5, len(sample_names) - 0.5)
    
    # Add legend
    if len(sig_columns) <= 15:  # Only show legend if not too many signatures
        ax.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left',
            fontsize=16,
            frameon=True,
            shadow=True
        )
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Remove padding and adjust layout to eliminate white space
    ax.margins(x=0, y=0)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2)
    
    return fig

def create_signature_heatmap(
    activities_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "CNA Signature Activities Heatmap"
) -> plt.Figure:
    """
    Create a heatmap of signature activities.
    
    Args:
        activities_df: DataFrame with samples as rows and signatures as columns
        figsize: Figure size (width, height)
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Get signature columns
    sig_columns = [col for col in activities_df.columns if 'signature' in col.lower() or 'sbs' in col.lower()]
    
    if not sig_columns:
        sig_columns = activities_df.columns.tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    data_for_heatmap = activities_df[sig_columns].T  # Transpose for better visualization
    
    im = ax.imshow(
        data_for_heatmap.values,
        cmap='YlOrRd',
        aspect='auto',
        interpolation='nearest'
    )
    
    # Set labels
    ax.set_xticks(range(len(data_for_heatmap.columns)))
    ax.set_xticklabels(data_for_heatmap.columns, rotation=45, ha='right')
    
    # Format y-axis labels with CON naming
    formatted_sig_names = format_signature_names(data_for_heatmap.index)
    ax.set_yticks(range(len(data_for_heatmap.index)))
    ax.set_yticklabels(formatted_sig_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Activity', rotation=270, labelpad=20, fontsize=18)
    
    # Set title
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_signature_summary_plot(
    activities_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Signature Activity Summary"
) -> plt.Figure:
    """
    Create a summary plot showing average signature activities.
    
    Args:
        activities_df: DataFrame with samples as rows and signatures as columns
        figsize: Figure size (width, height)
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Get signature columns
    sig_columns = [col for col in activities_df.columns if 'signature' in col.lower() or 'sbs' in col.lower()]
    
    if not sig_columns:
        sig_columns = activities_df.columns.tolist()
    
    # Calculate summary statistics
    means = activities_df[sig_columns].mean()
    stds = activities_df[sig_columns].std()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    x_pos = np.arange(len(sig_columns))
    bars = ax.bar(x_pos, means.values, yerr=stds.values, 
                  capsize=5, alpha=0.7, color='skyblue', 
                  edgecolor='navy', linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Signatures', fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean Activity ± SD', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Set x-axis labels with formatted names
    formatted_sig_names = format_signature_names(sig_columns)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(formatted_sig_names, rotation=45, ha='right')
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def save_plot_as_bytes(fig: plt.Figure, format: str = 'png', dpi: int = 300) -> bytes:
    """
    Save a matplotlib figure as bytes for download.
    
    Args:
        fig: matplotlib Figure object
        format: Output format ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats
        
    Returns:
        bytes: Figure data as bytes
    """
    buf = io.BytesIO()
    
    if format.lower() == 'pdf':
        fig.savefig(buf, format='pdf', bbox_inches='tight', dpi=dpi)
    elif format.lower() == 'svg':
        fig.savefig(buf, format='svg', bbox_inches='tight')
    else:  # Default to PNG
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    
    buf.seek(0)
    return buf.getvalue()

def create_bootstrap_confidence_plot(
    activities_df: pd.DataFrame,
    confidence_intervals: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Signature Activities with Confidence Intervals"
) -> plt.Figure:
    """
    Create a plot showing signature activities with bootstrap confidence intervals.
    
    Args:
        activities_df: DataFrame with samples as rows and signatures as columns
        confidence_intervals: DataFrame with confidence intervals (optional)
        figsize: Figure size (width, height)
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Get signature columns
    sig_columns = [col for col in activities_df.columns if 'signature' in col.lower() or 'sbs' in col.lower()]
    
    if not sig_columns:
        sig_columns = activities_df.columns.tolist()
    
    # Take first signature for demonstration
    if len(sig_columns) > 0:
        sig_name = sig_columns[0]
        formatted_sig_name = format_signature_names([sig_name])[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot main values
        x_pos = np.arange(len(activities_df))
        values = activities_df[sig_name].values
        
        ax.bar(x_pos, values, alpha=0.7, color='lightblue', 
               edgecolor='navy', linewidth=1, label='Activity')
        
        # Add confidence intervals if available
        if confidence_intervals is not None and f"{sig_name}_lower" in confidence_intervals.columns:
            lower = confidence_intervals[f"{sig_name}_lower"].values
            upper = confidence_intervals[f"{sig_name}_upper"].values
            
            # Add error bars
            ax.errorbar(x_pos, values, 
                       yerr=[values - lower, upper - values],
                       fmt='none', color='red', capsize=3, 
                       label='95% CI')
        
        # Customize plot
        ax.set_xlabel('Samples', fontsize=18, fontweight='bold')
        ax.set_ylabel(f'{formatted_sig_name} Activity', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        
        # Set x-axis labels
        sample_names = activities_df.index.tolist()
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sample_names, rotation=45, ha='right')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    else:
        # Return empty figure if no signatures found
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No signatures found', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=20, color='red')
        return fig

def create_quality_metrics_plot(
    r2_scores: np.ndarray,
    reconstruction_errors: np.ndarray,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Create a plot showing quality metrics (R² and reconstruction error).
    
    Args:
        r2_scores: Array of R² scores per sample
        reconstruction_errors: Array of reconstruction errors per sample
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # R² scores histogram
    ax1.hist(r2_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('R² Score', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Frequency', fontweight='bold', fontsize=18)
    ax1.set_title('R² Score Distribution', fontweight='bold', fontsize=20)
    ax1.axvline(np.mean(r2_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(r2_scores):.3f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Reconstruction error histogram
    ax2.hist(reconstruction_errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Reconstruction Error', fontweight='bold', fontsize=18)
    ax2.set_ylabel('Frequency', fontweight='bold', fontsize=18)
    ax2.set_title('Reconstruction Error Distribution', fontweight='bold', fontsize=20)
    ax2.axvline(np.mean(reconstruction_errors), color='blue', linestyle='--',
                label=f'Mean: {np.mean(reconstruction_errors):.6f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_advanced_visualization(
    activities_df: pd.DataFrame,
    plot_type: str = "stacked_bar",
    **kwargs
) -> plt.Figure:
    """
    Create advanced visualizations using the existing SignatureVisualizer if available.
    
    Args:
        activities_df: DataFrame with signature activities
        plot_type: Type of plot to create
        **kwargs: Additional arguments for the plot
        
    Returns:
        matplotlib Figure object
    """
    if SignatureVisualizer is None:
        # Fallback to basic plotting
        return create_stacked_bar_plot(activities_df, **kwargs)
    
    try:
        visualizer = SignatureVisualizer()
        
        if plot_type == "stacked_bar":
            fig = visualizer.plot_signature_contributions_stacked(
                activities_df, 
                **kwargs
            )
        elif plot_type == "heatmap":
            fig = visualizer.plot_signature_heatmap(
                activities_df,
                **kwargs
            )
        else:
            # Default to stacked bar
            fig = visualizer.plot_signature_contributions_stacked(
                activities_df,
                **kwargs
            )
        
        return fig
        
    except Exception as e:
        # Fallback to basic plotting on error
        print(f"Error using SignatureVisualizer: {e}")
        return create_stacked_bar_plot(activities_df, **kwargs)