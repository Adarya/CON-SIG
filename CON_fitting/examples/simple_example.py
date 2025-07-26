#!/usr/bin/env python3
"""
Simple example of using the CON_fitting framework.

This example demonstrates how to:
1. Load consensus signatures
2. Load CNA data (synthetic or from file)
3. Fit signature activities
4. Visualize results

Usage:
    python simple_example.py                           # Use synthetic data
    python simple_example.py your_data.csv             # Use real CSV data
    python simple_example.py your_data.tsv             # Use real TSV data
    python simple_example.py your_data.csv --transpose # Transpose if samples are columns
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processor import DataProcessor
from signature_fitter import ConsensusSignatureFitter
from visualizer import SignatureVisualizer


def detect_data_orientation(data_df, consensus_signatures):
    """
    Detect if samples are rows or columns by checking which orientation
    has more overlap with consensus signature categories.
    
    Args:
        data_df: Input DataFrame
        consensus_signatures: Consensus signatures DataFrame
        
    Returns:
        tuple: (needs_transpose, orientation_info)
    """
    cna_categories = set(consensus_signatures.index)
    
    # Check columns (samples as rows, CNA categories as columns)
    cols_overlap = len(set(data_df.columns) & cna_categories)
    cols_total = len(data_df.columns)
    cols_ratio = cols_overlap / cols_total if cols_total > 0 else 0
    
    # Check index (samples as columns, CNA categories as rows)  
    index_overlap = len(set(data_df.index) & cna_categories)
    index_total = len(data_df.index)
    index_ratio = index_overlap / index_total if index_total > 0 else 0
    
    print(f"   Data orientation analysis:")
    print(f"   - Columns as CNA categories: {cols_overlap}/{cols_total} match ({cols_ratio:.1%})")
    print(f"   - Index as CNA categories: {index_overlap}/{index_total} match ({index_ratio:.1%})")
    
    # If index has better overlap, we need to transpose
    needs_transpose = index_ratio > cols_ratio
    
    if needs_transpose:
        print(f"   → Detected: Samples are COLUMNS (will transpose)")
        return True, f"samples_as_columns"
    else:
        print(f"   → Detected: Samples are ROWS (no transpose needed)")
        return False, f"samples_as_rows"


def load_real_data(file_path, transpose_flag, consensus_signatures):
    """Load and validate real CNA data from file."""
    print(f"3. Loading real CNA data from: {file_path}")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data based on file extension
    if file_path.suffix.lower() == '.csv':
        data_df = pd.read_csv(file_path, index_col=0)
    elif file_path.suffix.lower() in ['.tsv', '.txt']:
        data_df = pd.read_csv(file_path, sep='\t', index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"   Loaded data with shape: {data_df.shape}")
    
    # Handle transpose decision
    if transpose_flag is None:
        # Auto-detect orientation
        needs_transpose, orientation = detect_data_orientation(data_df, consensus_signatures)
    else:
        # User specified
        needs_transpose = transpose_flag
        orientation = "user_specified"
        if needs_transpose:
            print(f"   → User specified: Transposing data (samples as columns)")
        else:
            print(f"   → User specified: No transpose (samples as rows)")
    
    # Transpose if needed
    if needs_transpose:
        print(f"   Transposing data...")
        data_df = data_df.T
        print(f"   New shape after transpose: {data_df.shape}")
    
    return data_df


def create_synthetic_data(consensus_signatures):
    """Create synthetic CNA data for demonstration."""
    print("3. Creating synthetic CNA data...")
    
    # For this example, we'll create synthetic data
    np.random.seed(42)
    n_samples = 10
    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    
    # Create synthetic CNA data that should fit well with the consensus signatures
    # This simulates realistic CNA data
    example_activities = np.random.exponential(scale=20, size=(n_samples, 5))
    example_activities[:, 1] = np.random.exponential(scale=5, size=n_samples)  # Lower activity for signature 2
    
    # Reconstruct CNA data from activities and signatures
    cna_data = example_activities @ consensus_signatures.values.T
    
    # Add some noise
    noise = np.random.normal(0, 0.1 * np.std(cna_data), cna_data.shape)
    cna_data += noise
    cna_data = np.maximum(cna_data, 0)  # Ensure non-negative
    
    # Create DataFrame
    cna_df = pd.DataFrame(
        cna_data,
        index=sample_names,
        columns=consensus_signatures.index
    )
    
    print(f"   Created {len(cna_df)} samples with {len(cna_df.columns)} CNA categories")
    return cna_df


def main(input_file=None, transpose=None):
    """Simple example of signature fitting."""
    print("CON_fitting Simple Example")
    print("=" * 40)
    
    # 1. Initialize components
    print("1. Initializing components...")
    processor = DataProcessor(verbose=True)
    visualizer = SignatureVisualizer()
    
    # 2. Load consensus signatures
    print("2. Loading consensus signatures...")
    consensus_signatures = processor.load_consensus_signatures('../data/consensus_signatures.csv')
    print(f"   Loaded {consensus_signatures.shape[1]} signatures with {consensus_signatures.shape[0]} CNA categories")
    
    # 3. Load CNA data (real or synthetic)
    if input_file:
        cna_df = load_real_data(input_file, transpose, consensus_signatures)
    else:
        cna_df = create_synthetic_data(consensus_signatures)
    
    # 4. Preprocess data
    print("4. Preprocessing data...")
    processed_data = processor.preprocess_cna_data(
        cna_df,
        reference_categories=consensus_signatures.index
    )
    
    # 5. Fit signatures using NNLS
    print("5. Fitting consensus signatures...")
    fitter = ConsensusSignatureFitter(
        consensus_signatures=consensus_signatures,
        method='nnls',
        verbose=True
    )
    
    fitted_activities, metrics = fitter.fit(processed_data)
    
    print(f"   Fitting completed with mean R² = {metrics['mean_r2']:.3f}")
    
    # 6. Display results
    print("6. Results:")
    print("\nFitted Activities:")
    print(fitted_activities.round(2))
    
    print(f"\nQuality Metrics:")
    print(f"   Mean R²: {metrics['mean_r2']:.3f}")
    print(f"   Mean Reconstruction Error: {metrics['mean_reconstruction_error']:.3f}")
    
    # 7. Create visualizations
    print("7. Creating visualizations...")
    
    # Determine output directory based on input
    if input_file:
        file_stem = Path(input_file).stem
        output_dir = Path(f'../output/real_data_{file_stem}')
    else:
        output_dir = Path('../output/example')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Activity heatmap
    fig1 = visualizer.plot_signature_activities(
        fitted_activities,
        output_path=output_dir / "fitted_activities.png",
        title=f"Fitted Signature Activities ({len(fitted_activities)} samples)",
        sample_labels=len(fitted_activities) <= 50  # Only show labels for small datasets
    )
    
    # Quality metrics
    fig2 = visualizer.plot_quality_metrics(
        metrics,
        output_path=output_dir / "quality_metrics.png",
        title="Fitting Quality Metrics"
    )
    
    # Signature contributions stacked bar plot
    fig3 = visualizer.plot_signature_contributions_stacked(
        fitted_activities,
        output_path=output_dir / "signature_contributions.pdf",
        title=f"Signature Contributions per Sample ({len(fitted_activities)} samples)",
        max_samples_per_plot=100
    )
    
    # Close figures
    visualizer.close_all_figures()
    
    print(f"   Plots saved to: {output_dir}")
    
    # 8. Save results
    print("8. Saving results...")
    fitted_activities.to_csv(output_dir / "fitted_activities.csv")
    processed_data.to_csv(output_dir / "processed_cna_data.csv")
    
    # Save summary
    summary_info = {
        'input_file': str(input_file) if input_file else 'synthetic',
        'n_samples': len(fitted_activities),
        'n_signatures': len(fitted_activities.columns),
        'mean_r2': metrics['mean_r2'],
        'mean_reconstruction_error': metrics['mean_reconstruction_error'],
        'transpose_applied': transpose if transpose is not None else 'auto-detected'
    }
    
    summary_df = pd.DataFrame([summary_info])
    summary_df.to_csv(output_dir / "analysis_summary.csv", index=False)
    
    print("\nExample completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    return fitted_activities, metrics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CON_fitting Simple Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python simple_example.py                           # Use synthetic data
    python simple_example.py data.csv                  # Auto-detect orientation
    python simple_example.py data.tsv --transpose      # Force transpose
    python simple_example.py data.csv --no-transpose   # Force no transpose
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default=None,
        help='Input CNA data file (CSV or TSV). If not provided, synthetic data will be used.'
    )
    
    transpose_group = parser.add_mutually_exclusive_group()
    transpose_group.add_argument(
        '--transpose',
        action='store_true',
        help='Force transpose the input data (use when samples are columns)'
    )
    transpose_group.add_argument(
        '--no-transpose',
        action='store_true',
        help='Force no transpose (use when samples are rows)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Determine transpose flag
    if args.transpose:
        transpose_flag = True
    elif args.no_transpose:
        transpose_flag = False
    else:
        transpose_flag = None  # Auto-detect
    
    try:
        activities, metrics = main(
            input_file=args.input_file,
            transpose=transpose_flag
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that your file exists and is readable")
        print("2. Ensure the file is in CSV or TSV format")
        print("3. Try using --transpose if samples are in columns")
        print("4. Check that CNA category names match the consensus signatures")
        sys.exit(1) 