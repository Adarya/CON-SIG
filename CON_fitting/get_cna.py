# cna_data_processing.py

import os
import pandas as pd
import numpy as np

def process_cna(raw_cna_df):
    """
    Process raw CNA DataFrame from cbioportal format to create processed CNA DataFrame 
    with engineered features compatible with FACETS format.

    Parameters:
        raw_cna_df (DataFrame): DataFrame containing raw CNA data with columns:
                                ID, chrom, loc.start, loc.end, num.mark, seg.mean

    Returns:
        cna_df (DataFrame): Processed CNA DataFrame with FACETS-compatible columns:
                           sample, chr, start, end, tcn.em, lcn.em
    """
    if raw_cna_df.empty:
        print("Raw CNA DataFrame is empty. Skipping processing.")
        return pd.DataFrame()

    print(f"Processing raw CNA DataFrame with {len(raw_cna_df)} segments...")

    # Create a copy to avoid modifying the original
    df = raw_cna_df.copy()
    
    # Map cbioportal column names to standard names
    column_mapping = {
        'ID': 'sample',
        'chrom': 'chr', 
        'loc.start': 'start',
        'loc.end': 'end',
        'seg.mean': 'segmentMean'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    required_cols = ['sample', 'chr', 'start', 'end', 'segmentMean']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("Calculating Total Copy Number (TCN) from segmentMean...")
    # Reverse the log2 transformation to get the Total Copy Number (TCN)
    # seg.mean is log2(copy_number/2), so copy_number = 2^(seg.mean) * 2
    df['tcn'] = 2 ** df['segmentMean'] * 2

    # Add tcn.em column (rounded TCN) - this is what FACETS expects
    df['tcn.em'] = np.round(df['tcn']).astype(int)

    # Ensure TCN is non-negative
    df['tcn.em'] = np.maximum(df['tcn.em'], 0)

    print("Calculating Lesser Copy Number (LCN) estimates...")
    # Add lcn.em column (lesser copy number estimate)
    # For segments without allele-specific information, we estimate:
    # - If TCN = 0: LCN = 0 (homozygous deletion)
    # - If TCN = 1: LCN = 0 (heterozygous deletion/LOH)  
    # - If TCN >= 2: LCN = 1 (assume heterozygous, this is a simplification)
    def estimate_lcn(tcn):
        if tcn <= 1:
            return 0
        else:
            return 1

    df['lcn.em'] = df['tcn.em'].apply(estimate_lcn)

    # Create TCN categories for reporting
    def categorize_tcn(tcn):
        if tcn == 0:
            return 'Homozygous Deletion'
        elif tcn == 1:
            return 'Heterozygous Deletion'  
        elif tcn == 2:
            return 'Normal'
        elif tcn in [3, 4]:
            return 'Gain'
        elif tcn in [5, 6, 7, 8]:
            return 'Amplification'
        else:
            return 'High Amplification'
    
    df['tcn_category'] = df['tcn.em'].apply(categorize_tcn)

    # Select and reorder columns for FACETS format
    facets_columns = ['sample', 'chr', 'start', 'end', 'tcn.em', 'lcn.em']
    cna_df = df[facets_columns].copy()

    print("Processing completed!")
    print(f"Processed {len(cna_df)} segments from {cna_df['sample'].nunique()} unique samples")

    # Report statistics
    print("\nTCN.EM distribution:")
    print(df['tcn.em'].value_counts().sort_index())

    print("\nLCN.EM distribution:")
    print(df['lcn.em'].value_counts().sort_index())

    print("\nTCN category distribution:")
    print(df['tcn_category'].value_counts())

    print(f"\nSample names (first 10): {list(cna_df['sample'].unique()[:10])}")
    
    print("\nProcessed DataFrame (first 5 rows):")
    print(cna_df.head())

    return cna_df


def load_cbioportal_segments(file_path):
    """
    Load CNA segments from cbioportal format file.
    
    Parameters:
        file_path (str): Path to the cbioportal CNA segments file
        
    Returns:
        DataFrame: Raw CNA segments
    """
    print(f"Loading cbioportal CNA segments from: {file_path}")
    
    # Load the file - it's tab-separated
    raw_df = pd.read_csv(file_path, sep='\t')
    
    print(f"Loaded {len(raw_df)} segments")
    print(f"Columns: {list(raw_df.columns)}")
    print(f"Number of samples: {raw_df['ID'].nunique()}")
    
    return raw_df


def process_cbioportal_to_facets(input_file, output_file=None):
    """
    Complete pipeline to process cbioportal CNA segments to FACETS format.
    
    Parameters:
        input_file (str): Path to cbioportal CNA segments file
        output_file (str, optional): Path to save processed file. If None, auto-generated.
        
    Returns:
        DataFrame: Processed CNA data in FACETS format
    """
    # Load raw data
    raw_df = load_cbioportal_segments(input_file)
    
    # Process to FACETS format
    processed_df = process_cna(raw_df)
    
    # Save if output file specified
    if output_file:
        processed_df.to_csv(output_file, sep='\t', index=False)
        print(f"\nProcessed data saved to: {output_file}")
    
    return processed_df


if __name__ == "__main__":
    # Test with the rectal MSK data
    input_file = "/Users/adary/Documents/cbioportal_data/rectal_msk_2022/data_cna_hg19.seg"
    output_file = "processed_cna_segments_facets.tsv"
    
    try:
        processed_data = process_cbioportal_to_facets(input_file, output_file)
        print(f"\nProcessing completed successfully!")
        print(f"Output shape: {processed_data.shape}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()