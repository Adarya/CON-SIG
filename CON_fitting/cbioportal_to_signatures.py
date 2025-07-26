#!/usr/bin/env python3
"""
Complete One-Click Pipeline: cbioportal CNA Segments → Signature Activities

This pipeline provides end-to-end analysis from raw cbioportal CNA segments 
to consensus signature activities with comprehensive visualization and results.

Pipeline Steps:
1. Load raw cbioportal CNA segments (.seg format)
2. Process to FACETS-compatible format (TCN/LCN calculation)
3. Generate CNV28 matrix using CNVMatrixGenerator
4. Fit consensus signatures using CON_fitting framework
5. Create comprehensive visualizations
6. Save all results in project-specific directory

Usage:
    python cbioportal_to_signatures.py input_file project_name [output_dir]
    
Example:
    python cbioportal_to_signatures.py data_cna_hg19.seg RectalMSK2022
    python cbioportal_to_signatures.py /path/to/data_cna_hg19.seg MyProject ./results/
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add src to path for CON_fitting framework
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our modules
from get_cna import process_cbioportal_to_facets
from CNVMatrixGenerator import generateCNVMatrix
from data_processor import DataProcessor
from signature_fitter import ConsensusSignatureFitter
from visualizer import SignatureVisualizer


def print_step(step_num, title, details=""):
    """Print formatted step information."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")
    if details:
        print(f"{details}")
    print()


def create_project_directory(project_name, base_output_dir="./results/"):
    """Create project-specific directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(base_output_dir) / f"{project_name}_{timestamp}"
    
    # Create subdirectories
    subdirs = {
        'processed_data': project_dir / 'processed_data',
        'matrices': project_dir / 'matrices', 
        'signatures': project_dir / 'signatures',
        'visualizations': project_dir / 'visualizations',
        'summary': project_dir / 'summary'
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return project_dir, subdirs


def load_consensus_signatures():
    """Load consensus signatures for fitting."""
    consensus_file = Path(__file__).parent.parent / 'results' / 'consensus_signatures.csv'
    
    if not consensus_file.exists():
        raise FileNotFoundError(f"Consensus signatures not found: {consensus_file}")
    
    signatures_df = pd.read_csv(consensus_file, index_col=0)
    print(f"   ✓ Loaded {signatures_df.shape[1]} consensus signatures")
    print(f"   ✓ Features: {signatures_df.shape[0]} CNA categories")
    
    return signatures_df


def process_cna_segments(input_file, project_name, processed_data_dir):
    """Step 1: Process cbioportal CNA segments to FACETS format."""
    print_step(1, "Processing cbioportal CNA Segments", 
              f"Converting {input_file} to FACETS format")
    
    # Define output file
    processed_segments_file = processed_data_dir / f"{project_name}_processed_segments.tsv"
    
    try:
        # Process using our existing function
        processed_data = process_cbioportal_to_facets(input_file, str(processed_segments_file))
        
        print(f"   ✓ Successfully processed {len(processed_data)} segments")
        print(f"   ✓ Number of samples: {processed_data['sample'].nunique()}")
        print(f"   ✓ Output saved: {processed_segments_file}")
        
        return str(processed_segments_file), processed_data
        
    except Exception as e:
        print(f"   ✗ Error in CNA processing: {str(e)}")
        raise


def generate_cnv_matrix(processed_segments_file, project_name, matrices_dir):
    """Step 2: Generate CNV28 matrix from processed segments."""
    print_step(2, "Generating CNV Matrix", 
              "Creating CNV28 matrix for signature analysis")
    
    try:
        # Generate matrix using CNVMatrixGenerator
        matrix_result = generateCNVMatrix(
            file_type="FACETS",
            input_file=processed_segments_file,
            project=project_name,
            output_path=str(matrices_dir),
            folder=False
        )
        
        # Load the generated matrix
        matrix_file = matrices_dir / f"{project_name}.CNV28.matrix.tsv"
        cnv_matrix = pd.read_csv(matrix_file, sep='\t', index_col=0)
        
        print(f"   ✓ Generated CNV matrix: {cnv_matrix.shape}")
        print(f"   ✓ Features: {cnv_matrix.shape[0]} (28 CNA categories)")
        print(f"   ✓ Samples: {cnv_matrix.shape[1]}")
        print(f"   ✓ Total CNA events: {cnv_matrix.values.sum():,}")
        print(f"   ✓ Matrix saved: {matrix_file}")
        
        return str(matrix_file), cnv_matrix
        
    except Exception as e:
        print(f"   ✗ Error in matrix generation: {str(e)}")
        raise


def fit_consensus_signatures(cnv_matrix, consensus_signatures, project_name, signatures_dir):
    """Step 3: Fit consensus signatures to CNV matrix."""
    print_step(3, "Fitting Consensus Signatures", 
              "Applying CON_fitting framework for signature analysis")
    
    try:
        # Initialize components
        processor = DataProcessor(verbose=True)
        
        # Transpose matrix so samples are rows (CON_fitting expects this format)
        cna_data = cnv_matrix.T
        print(f"   Data format: {cna_data.shape} (samples × features)")
        
        # Preprocess data
        print("   Preprocessing CNA data...")
        processed_data = processor.preprocess_cna_data(
            cna_data,
            reference_categories=consensus_signatures.index
        )
        
        # Initialize fitter
        fitter = ConsensusSignatureFitter(
            consensus_signatures=consensus_signatures,
            method='nnls',
            verbose=True
        )
        
        # Fit signatures
        print("   Fitting signatures using NNLS...")
        fitted_activities, metrics = fitter.fit(processed_data)
        
        # Save results
        activities_file = signatures_dir / f"{project_name}_signature_activities.csv"
        fitted_activities.to_csv(activities_file)
        
        processed_data_file = signatures_dir / f"{project_name}_processed_cnv_data.csv"
        processed_data.to_csv(processed_data_file)
        
        metrics_file = signatures_dir / f"{project_name}_fitting_metrics.csv"
        metrics_df = pd.DataFrame([{
            'mean_r2': metrics['mean_r2'],
            'mean_reconstruction_error': metrics['mean_reconstruction_error'],
            'n_samples': len(fitted_activities),
            'n_signatures': len(fitted_activities.columns)
        }])
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"   ✓ Signature fitting completed!")
        print(f"   ✓ Mean R²: {metrics['mean_r2']:.3f}")
        print(f"   ✓ Mean reconstruction error: {metrics['mean_reconstruction_error']:.3f}")
        print(f"   ✓ Activities saved: {activities_file}")
        print(f"   ✓ Metrics saved: {metrics_file}")
        
        return fitted_activities, metrics, processed_data
        
    except Exception as e:
        print(f"   ✗ Error in signature fitting: {str(e)}")
        raise


def create_visualizations(fitted_activities, metrics, processed_data, project_name, visualizations_dir):
    """Step 4: Create comprehensive visualizations."""
    print_step(4, "Creating Visualizations", 
              "Generating plots and visual summaries")
    
    try:
        visualizer = SignatureVisualizer()
        
        # 1. Signature activities heatmap
        print("   Creating signature activities heatmap...")
        activities_plot = visualizations_dir / f"{project_name}_signature_activities.png"
        fig1 = visualizer.plot_signature_activities(
            fitted_activities,
            output_path=str(activities_plot),
            title=f"Consensus Signature Activities - {project_name}",
            sample_labels=len(fitted_activities) <= 50
        )
        
        # 2. Quality metrics plot
        print("   Creating quality metrics visualization...")
        metrics_plot = visualizations_dir / f"{project_name}_quality_metrics.png"
        fig2 = visualizer.plot_quality_metrics(
            metrics,
            output_path=str(metrics_plot),
            title=f"Signature Fitting Quality - {project_name}"
        )
        
        # 3. Signature contributions stacked bar plot
        print("   Creating signature contributions plot...")
        contributions_plot = visualizations_dir / f"{project_name}_signature_contributions.pdf"
        fig3 = visualizer.plot_signature_contributions_stacked(
            fitted_activities,
            output_path=str(contributions_plot),
            title=f"Signature Contributions per sample - {'Sarcoma' if project_name == 'sarc_msk' else project_name}",
            max_samples_per_plot=100
        )
        
        # Close figures to free memory
        visualizer.close_all_figures()
        
        print(f"   ✓ Activities heatmap: {activities_plot}")
        print(f"   ✓ Quality metrics: {metrics_plot}")
        print(f"   ✓ Contributions plot: {contributions_plot}")
        
        return [str(activities_plot), str(metrics_plot), str(contributions_plot)]
        
    except Exception as e:
        print(f"   ✗ Error in visualization: {str(e)}")
        raise


def create_summary_report(input_file, project_name, fitted_activities, metrics, 
                         processed_data, project_dir, summary_dir):
    """Step 5: Create comprehensive summary report."""
    print_step(5, "Creating Summary Report", 
              "Generating comprehensive analysis summary")
    
    try:
        # Calculate summary statistics
        n_samples = len(fitted_activities)
        n_signatures = len(fitted_activities.columns)
        total_activity = fitted_activities.sum().sum()
        
        # Signature statistics
        signature_stats = fitted_activities.describe()
        dominant_signatures = fitted_activities.mean().sort_values(ascending=False)
        
        # Sample statistics
        sample_total_activity = fitted_activities.sum(axis=1)
        
        # Create summary report
        summary_file = summary_dir / f"{project_name}_analysis_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"CNA Signature Analysis Summary Report\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Project: {project_name}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input File: {input_file}\n")
            f.write(f"Output Directory: {project_dir}\n\n")
            
            f.write(f"Dataset Overview:\n")
            f.write(f"  - Number of samples: {n_samples:,}\n")
            f.write(f"  - Number of signatures: {n_signatures}\n")
            f.write(f"  - Total signature activity: {total_activity:.1f}\n\n")
            
            f.write(f"Fitting Quality:\n")
            f.write(f"  - Mean R²: {metrics['mean_r2']:.3f}\n")
            f.write(f"  - Mean reconstruction error: {metrics['mean_reconstruction_error']:.3f}\n\n")
            
            f.write(f"Signature Activity Summary (mean ± std):\n")
            for sig in dominant_signatures.index:
                mean_val = dominant_signatures[sig]
                std_val = fitted_activities[sig].std()
                f.write(f"  - {sig}: {mean_val:.2f} ± {std_val:.2f}\n")
            f.write(f"\n")
            
            f.write(f"Sample Activity Range:\n")
            f.write(f"  - Minimum total activity: {sample_total_activity.min():.2f}\n")
            f.write(f"  - Maximum total activity: {sample_total_activity.max():.2f}\n")
            f.write(f"  - Mean total activity: {sample_total_activity.mean():.2f}\n")
            f.write(f"  - Median total activity: {sample_total_activity.median():.2f}\n\n")
            
            f.write(f"Files Generated:\n")
            f.write(f"  - Processed segments: processed_data/{project_name}_processed_segments.tsv\n")
            f.write(f"  - CNV matrix: matrices/{project_name}.CNV28.matrix.tsv\n")
            f.write(f"  - Signature activities: signatures/{project_name}_signature_activities.csv\n")
            f.write(f"  - Fitting metrics: signatures/{project_name}_fitting_metrics.csv\n")
            f.write(f"  - Visualizations: visualizations/ (3 PNG files)\n")
            f.write(f"  - This summary: summary/{project_name}_analysis_summary.txt\n")
        
        # Save detailed signature statistics
        signature_stats_file = summary_dir / f"{project_name}_signature_statistics.csv"
        signature_stats.to_csv(signature_stats_file)
        
        # Save sample statistics
        sample_stats_file = summary_dir / f"{project_name}_sample_statistics.csv"
        sample_stats = pd.DataFrame({
            'sample': fitted_activities.index,
            'total_activity': sample_total_activity,
            'dominant_signature': fitted_activities.idxmax(axis=1),
            'dominant_signature_activity': fitted_activities.max(axis=1)
        })
        sample_stats.to_csv(sample_stats_file, index=False)
        
        print(f"   ✓ Summary report: {summary_file}")
        print(f"   ✓ Signature statistics: {signature_stats_file}")
        print(f"   ✓ Sample statistics: {sample_stats_file}")
        
        return str(summary_file)
        
    except Exception as e:
        print(f"   ✗ Error creating summary: {str(e)}")
        raise


def main(input_file, project_name, output_dir="./results/"):
    """Main pipeline function."""
    start_time = time.time()
    
    print(f"\n🚀 cbioportal → Signature Analysis Pipeline")
    print(f"{'='*80}")
    print(f"Project: {project_name}")
    print(f"Input: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup project directory
        project_dir, subdirs = create_project_directory(project_name, output_dir)
        print(f"📁 Project directory: {project_dir}")
        
        # Load consensus signatures
        consensus_signatures = load_consensus_signatures()
        
        # Step 1: Process CNA segments
        processed_segments_file, processed_segments_data = process_cna_segments(
            input_file, project_name, subdirs['processed_data']
        )
        
        # Step 2: Generate CNV matrix
        matrix_file, cnv_matrix = generate_cnv_matrix(
            processed_segments_file, project_name, subdirs['matrices']
        )
        
        # Step 3: Fit consensus signatures
        fitted_activities, metrics, processed_data = fit_consensus_signatures(
            cnv_matrix, consensus_signatures, project_name, subdirs['signatures']
        )
        
        # Step 4: Create visualizations
        plot_files = create_visualizations(
            fitted_activities, metrics, processed_data, project_name, subdirs['visualizations']
        )
        
        # Step 5: Create summary report
        summary_file = create_summary_report(
            input_file, project_name, fitted_activities, metrics, processed_data,
            project_dir, subdirs['summary']
        )
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print_step("✅", "PIPELINE COMPLETED SUCCESSFULLY!", 
                  f"Total time: {elapsed_time:.1f} seconds")
        
        print(f"📊 RESULTS SUMMARY:")
        print(f"   • Samples analyzed: {len(fitted_activities):,}")
        print(f"   • Signatures fitted: {len(fitted_activities.columns)}")
        print(f"   • Mean fitting R²: {metrics['mean_r2']:.3f}")
        print(f"   • Output directory: {project_dir}")
        
        print(f"\n📁 Generated Files:")
        print(f"   └── {project_dir.name}/")
        print(f"       ├── processed_data/     (FACETS-format segments)")
        print(f"       ├── matrices/           (CNV28 matrix)")
        print(f"       ├── signatures/         (fitted activities & metrics)")
        print(f"       ├── visualizations/     (3 publication-ready plots)")
        print(f"       └── summary/            (comprehensive reports)")
        
        print(f"\n🎉 Ready for downstream analysis and publication!")
        
        return {
            'project_dir': str(project_dir),
            'fitted_activities': fitted_activities,
            'metrics': metrics,
            'cnv_matrix': cnv_matrix,
            'plot_files': plot_files,
            'summary_file': summary_file
        }
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED")
        print(f"Error: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"1. Check input file format and permissions")
        print(f"2. Ensure consensus signatures are available")
        print(f"3. Verify sufficient disk space")
        print(f"4. Check Python package dependencies")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete cbioportal CNA Segments → Signature Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cbioportal_to_signatures.py data_cna_hg19.seg RectalMSK2022
    python cbioportal_to_signatures.py /path/to/segments.seg MyProject ./my_results/
    
This pipeline will:
1. Process cbioportal segments to FACETS format
2. Generate CNV28 matrix for signature analysis  
3. Fit consensus signatures using CON_fitting framework
4. Create comprehensive visualizations
5. Generate detailed summary reports

Output structure:
    project_name_TIMESTAMP/
    ├── processed_data/     # FACETS-compatible segments
    ├── matrices/           # CNV28 matrix 
    ├── signatures/         # Fitted activities and metrics
    ├── visualizations/     # Publication-ready plots
    └── summary/            # Analysis reports and statistics
        """
    )
    
    parser.add_argument(
        'input_file',
        help='cbioportal CNA segments file (.seg format with ID, chrom, loc.start, loc.end, seg.mean columns)'
    )
    
    parser.add_argument(
        'project_name',
        help='Project name for output files and directories'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='./results/',
        help='Base output directory (default: ./results/)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        results = main(
            input_file=args.input_file,
            project_name=args.project_name,
            output_dir=args.output_dir
        )
        sys.exit(0)
        
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {str(e)}")
        sys.exit(1) 