#!/usr/bin/env python3
"""
Complete pipeline to process cbioportal CNA segments and generate CNV matrix.

This script:
1. Loads raw CNA segments from cbioportal format 
2. Processes them to FACETS-compatible format (adds TCN/LCN features)
3. Generates CNV matrix using CNVMatrixGenerator
4. Outputs both processed segments and CNV matrix

Usage:
    python cbioportal_pipeline.py input_file project_name [output_dir]
    
Example:
    python cbioportal_pipeline.py /path/to/data_cna_hg19.seg MyProject ./output/
"""

import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# Import our modules
from get_cna import process_cbioportal_to_facets
from CNVMatrixGenerator import generateCNVMatrix


def main(input_file, project_name, output_dir="./output/"):
    """
    Main pipeline function.
    
    Args:
        input_file (str): Path to cbioportal CNA segments file
        project_name (str): Name for the project/output files
        output_dir (str): Directory to save outputs
    """
    print(f"=== cbioportal to CNV Matrix Pipeline ===")
    print(f"Input file: {input_file}")
    print(f"Project name: {project_name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process cbioportal segments to FACETS format
    print("Step 1: Processing cbioportal segments to FACETS format...")
    processed_segments_file = os.path.join(output_dir, f"{project_name}_processed_segments.tsv")
    
    try:
        processed_data = process_cbioportal_to_facets(input_file, processed_segments_file)
        print(f"✓ Successfully processed {len(processed_data)} segments from {processed_data['sample'].nunique()} samples")
    except Exception as e:
        print(f"✗ Error in step 1: {str(e)}")
        return False
    
    print()
    
    # Step 2: Generate CNV matrix using CNVMatrixGenerator
    print("Step 2: Generating CNV matrix...")
    
    try:
        matrix_output_path = generateCNVMatrix(
            file_type="FACETS",
            input_file=processed_segments_file,
            project=project_name,
            output_path=output_dir,
            folder=False
        )
        print(f"✓ Successfully generated CNV matrix")
        
        # Find the generated matrix file
        expected_matrix_file = os.path.join(output_dir, f"{project_name}.CNV28.matrix.tsv")
        if os.path.exists(expected_matrix_file):
            # Load and report matrix statistics
            matrix_df = pd.read_csv(expected_matrix_file, sep='\t', index_col=0)
            print(f"✓ Matrix shape: {matrix_df.shape} (28 features × {matrix_df.shape[1]} samples)")
            print(f"✓ Total CNA events: {matrix_df.values.sum():,}")
            print(f"✓ Matrix file: {expected_matrix_file}")
        
    except Exception as e:
        print(f"✗ Error in step 2: {str(e)}")
        return False
    
    print()
    
    # Summary
    print("=== Pipeline Completed Successfully! ===")
    print(f"Files generated:")
    print(f"  1. Processed segments: {processed_segments_file}")
    print(f"  2. CNV matrix: {expected_matrix_file}")
    print()
    print("Next steps:")
    print("  - Use the CNV matrix for signature analysis")
    print("  - Apply the CON_fitting framework to extract signature activities")
    
    return True


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Process cbioportal CNA segments and generate CNV matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cbioportal_pipeline.py data_cna_hg19.seg RectalMSK2022
  python cbioportal_pipeline.py /path/to/data_cna_hg19.seg MyProject ./output/
  
The pipeline will:
1. Load cbioportal CNA segments (ID, chrom, loc.start, loc.end, seg.mean)
2. Calculate TCN (Total Copy Number) and LCN (Lesser Copy Number) 
3. Generate CNV28 matrix compatible with CON_fitting framework
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to cbioportal CNA segments file (.seg format)'
    )
    
    parser.add_argument(
        'project_name', 
        help='Project name for output files'
    )
    
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='./output/',
        help='Output directory (default: ./output/)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return False
    
    # Run pipeline
    success = main(args.input_file, args.project_name, args.output_dir)
    
    return success


if __name__ == "__main__":
    success = cli()
    sys.exit(0 if success else 1) 