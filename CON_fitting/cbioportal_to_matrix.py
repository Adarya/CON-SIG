#!/usr/bin/env python3
"""
Complete pipeline to process cbioportal CNA segments and generate CNV matrix.

This script:
1. Loads raw CNA segments from cbioportal format
2. Processes them to FACETS-compatible format
3. Generates CNV matrix using CNVMatrixGenerator
4. Outputs both processed segments and CNV matrix

Usage:
    python cbioportal_to_matrix.py input_file project_name output_dir
    python cbioportal_to_matrix.py /path/to/data_cna_hg19.seg MyProject ./output/
"""

import sys
import os
from pathlib import Path
import pandas as pd
import argparse

# Import our modules
from get_cna import process_cbioportal_to_facets
from CNVMatrixGenerator import generateCNVMatrix


def main(input_file, project_name, output_dir):
    """
    Main pipeline function.
    
    Args:
        input_file: Path to cbioportal CNA segments file
        project_name: Name for the project/output files
        output_dir: Directory to save outputs
    """
    print("="*60)
    print("CBIOPORTAL CNA SEGMENTS TO CNV MATRIX PIPELINE")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process cbioportal segments to FACETS format
    print("\nSTEP 1: Processing cbioportal segments to FACETS format")
    print("-" * 50)
    
    processed_file = output_path / f"{project_name}_processed_segments.tsv"
    
    try:
        processed_df = process_cbioportal_to_facets(
            input_file=input_file,
            output_file=str(processed_file)
        )
        
        print(f"✅ Step 1 completed successfully!")
        print(f"   Processed segments saved to: {processed_file}")
        print(f"   Shape: {processed_df.shape}")
        print(f"   Samples: {processed_df['sample'].nunique()}")
        
    except Exception as e:
        print(f"❌ Step 1 failed: {str(e)}")
        raise
    
    # Step 2: Generate CNV matrix using FACETS format
    print("\nSTEP 2: Generating CNV matrix using FACETS format")
    print("-" * 50)
    
    try:
        cnv_matrix = generateCNVMatrix(
            file_type="FACETS",
            input_file=str(processed_file),
            project=project_name,
            output_path=str(output_path)
        )
        
        print(f"✅ Step 2 completed successfully!")
        print(f"   CNV matrix saved to: {output_path / f'{project_name}.CNV48.matrix.tsv'}")
        print(f"   Matrix shape: {cnv_matrix.shape}")
        print(f"   Features: {cnv_matrix.shape[0]}")
        print(f"   Samples: {cnv_matrix.shape[1] - 1}")  # -1 for MutationType column
        
    except Exception as e:
        print(f"❌ Step 2 failed: {str(e)}")
        raise
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"Input file: {input_file}")
    print(f"Project name: {project_name}")
    print(f"Output directory: {output_path}")
    print(f"\nGenerated files:")
    print(f"  1. Processed segments: {processed_file}")
    print(f"  2. CNV matrix: {output_path / f'{project_name}.CNV48.matrix.tsv'}")
    
    return cnv_matrix, processed_df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process cbioportal CNA segments and generate CNV matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cbioportal_to_matrix.py data_cna_hg19.seg RectalMSK ./output/
    python cbioportal_to_matrix.py /path/to/segments.seg MyProject /output/dir/
    
The script will:
1. Process raw cbioportal segments to FACETS format
2. Generate CNV 48-feature matrix
3. Save both processed segments and matrix
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to cbioportal CNA segments file (data_cna_hg19.seg format)'
    )
    
    parser.add_argument(
        'project_name',
        help='Name for the project (used in output filenames)'
    )
    
    parser.add_argument(
        'output_dir',
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only validate input file format, do not process'
    )
    
    return parser.parse_args()


def validate_input_file(input_file):
    """Validate the input file format."""
    print(f"Validating input file: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Check file format
    try:
        df = pd.read_csv(input_file, sep='\t', nrows=5)
        expected_cols = ['ID', 'chrom', 'loc.start', 'loc.end', 'seg.mean']
        
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"   Found columns: {list(df.columns)}")
            return False
        
        print(f"✅ Input file format is valid")
        print(f"   Columns: {list(df.columns)}")
        print(f"   First few rows:")
        print(df.head())
        return True
        
    except Exception as e:
        print(f"❌ Error reading input file: {str(e)}")
        return False


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        # Validate input file
        if not validate_input_file(args.input_file):
            print("❌ Input file validation failed!")
            sys.exit(1)
        
        # If only checking, exit here
        if args.check_only:
            print("✅ Input file validation passed!")
            sys.exit(0)
        
        # Run the full pipeline
        cnv_matrix, processed_df = main(
            input_file=args.input_file,
            project_name=args.project_name,
            output_dir=args.output_dir
        )
        
        print("\n🎉 All done! Check the output directory for results.")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 