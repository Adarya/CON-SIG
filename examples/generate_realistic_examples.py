"""
Generate realistic example CNA matrix data based on the consensus signatures.
Creates samples with varied signature contributions for demonstration purposes.
"""

import pandas as pd
import numpy as np

# Feature names for the 28 CNA categories
FEATURE_NAMES = [
    '0:homdel:0-100kb', '0:homdel:100kb-1Mb', '0:homdel:>1Mb',
    '1:LOH:0-100kb', '1:LOH:100kb-1Mb', '1:LOH:1Mb-10Mb', '1:LOH:10Mb-40Mb', '1:LOH:>40Mb',
    '2:het:0-100kb', '2:het:100kb-1Mb', '2:het:1Mb-10Mb', '2:het:10Mb-40Mb', '2:het:>40Mb',
    '3-4:het:0-100kb', '3-4:het:100kb-1Mb', '3-4:het:1Mb-10Mb', '3-4:het:10Mb-40Mb', '3-4:het:>40Mb',
    '5-8:het:0-100kb', '5-8:het:100kb-1Mb', '5-8:het:1Mb-10Mb', '5-8:het:10Mb-40Mb', '5-8:het:>40Mb',
    '9+:het:0-100kb', '9+:het:100kb-1Mb', '9+:het:1Mb-10Mb', '9+:het:10Mb-40Mb', '9+:het:>40Mb'
]

# Approximate consensus signature profiles (based on manuscript descriptions)
# These represent the characteristic CNA patterns for each signature
CONSENSUS_SIGNATURES = {
    'CON1': {  # LOH + 3-4 copy gains, arm-level events
        '1:LOH:1Mb-10Mb': 0.15, '1:LOH:10Mb-40Mb': 0.20, '1:LOH:>40Mb': 0.10,
        '2:het:10Mb-40Mb': 0.10, '2:het:>40Mb': 0.05,
        '3-4:het:1Mb-10Mb': 0.15, '3-4:het:10Mb-40Mb': 0.15, '3-4:het:>40Mb': 0.10
    },
    'CON2': {  # Focal amplifications 5-8 copies (1-10 Mb)
        '5-8:het:100kb-1Mb': 0.20, '5-8:het:1Mb-10Mb': 0.40, '5-8:het:10Mb-40Mb': 0.15,
        '2:het:1Mb-10Mb': 0.10, '2:het:10Mb-40Mb': 0.10,
        '9+:het:100kb-1Mb': 0.03, '9+:het:1Mb-10Mb': 0.02
    },
    'CON3': {  # Large LOH segments
        '1:LOH:10Mb-40Mb': 0.35, '1:LOH:>40Mb': 0.35,
        '1:LOH:1Mb-10Mb': 0.10,
        '2:het:10Mb-40Mb': 0.10, '2:het:>40Mb': 0.10
    },
    'CON4': {  # Mixed - LOH, het, low-level amp
        '1:LOH:1Mb-10Mb': 0.10, '1:LOH:10Mb-40Mb': 0.12, '1:LOH:>40Mb': 0.08,
        '2:het:1Mb-10Mb': 0.15, '2:het:10Mb-40Mb': 0.15, '2:het:>40Mb': 0.05,
        '3-4:het:100kb-1Mb': 0.10, '3-4:het:1Mb-10Mb': 0.12, '3-4:het:10Mb-40Mb': 0.08,
        '0:homdel:100kb-1Mb': 0.03, '0:homdel:>1Mb': 0.02
    },
    'CON5': {  # Near-diploid (mostly 2:het)
        '2:het:1Mb-10Mb': 0.20, '2:het:10Mb-40Mb': 0.30, '2:het:>40Mb': 0.35,
        '2:het:100kb-1Mb': 0.10,
        '3-4:het:>40Mb': 0.05
    }
}

def generate_sample(signature_weights, noise_level=0.05):
    """Generate a sample as a weighted combination of signatures plus noise."""
    sample = np.zeros(28)

    for sig_name, weight in signature_weights.items():
        sig_profile = np.zeros(28)
        sig_dict = CONSENSUS_SIGNATURES[sig_name]
        for feat, val in sig_dict.items():
            idx = FEATURE_NAMES.index(feat)
            sig_profile[idx] = val

        sample += weight * sig_profile

    # Add noise
    noise = np.abs(np.random.normal(0, noise_level, 28))
    sample += noise

    # Normalize to sum to 1
    sample = sample / sample.sum()

    return sample

def generate_example_dataset(n_samples=40):
    """Generate a diverse example dataset with varied signature profiles."""
    np.random.seed(42)  # For reproducibility

    samples = []
    sample_names = []
    cancer_types = []

    # Group 1: CON5-dominant (diploid-like, 10 samples)
    for i in range(10):
        weights = {'CON5': np.random.uniform(0.6, 0.9)}
        remaining = 1 - weights['CON5']
        weights['CON4'] = remaining * np.random.uniform(0.3, 0.6)
        weights['CON1'] = remaining * np.random.uniform(0.1, 0.3)
        weights['CON3'] = remaining * np.random.uniform(0.05, 0.2)
        weights['CON2'] = remaining - weights['CON4'] - weights['CON1'] - weights['CON3']
        samples.append(generate_sample(weights))
        sample_names.append(f'Sample_Diploid_{i+1:02d}')
        cancer_types.append('BRCA' if i < 5 else 'CRC')

    # Group 2: CON1-dominant (LOH + gains, 8 samples)
    for i in range(8):
        weights = {'CON1': np.random.uniform(0.4, 0.7)}
        remaining = 1 - weights['CON1']
        weights['CON4'] = remaining * np.random.uniform(0.3, 0.5)
        weights['CON5'] = remaining * np.random.uniform(0.2, 0.4)
        weights['CON3'] = remaining * np.random.uniform(0.1, 0.2)
        weights['CON2'] = remaining - weights['CON4'] - weights['CON5'] - weights['CON3']
        samples.append(generate_sample(weights))
        sample_names.append(f'Sample_LOH_Gain_{i+1:02d}')
        cancer_types.append('PRAD' if i < 4 else 'NSCLC')

    # Group 3: CON4-dominant (mixed aneuploid, 8 samples)
    for i in range(8):
        weights = {'CON4': np.random.uniform(0.4, 0.7)}
        remaining = 1 - weights['CON4']
        weights['CON1'] = remaining * np.random.uniform(0.2, 0.4)
        weights['CON5'] = remaining * np.random.uniform(0.2, 0.4)
        weights['CON3'] = remaining * np.random.uniform(0.1, 0.2)
        weights['CON2'] = max(0, remaining - weights['CON1'] - weights['CON5'] - weights['CON3'])
        samples.append(generate_sample(weights))
        sample_names.append(f'Sample_Mixed_{i+1:02d}')
        cancer_types.append('PAAD' if i < 4 else 'NSCLC')

    # Group 4: CON3-dominant (large LOH, 6 samples)
    for i in range(6):
        weights = {'CON3': np.random.uniform(0.4, 0.6)}
        remaining = 1 - weights['CON3']
        weights['CON1'] = remaining * np.random.uniform(0.2, 0.4)
        weights['CON4'] = remaining * np.random.uniform(0.2, 0.3)
        weights['CON5'] = remaining * np.random.uniform(0.1, 0.3)
        weights['CON2'] = max(0, remaining - weights['CON1'] - weights['CON4'] - weights['CON5'])
        samples.append(generate_sample(weights))
        sample_names.append(f'Sample_LOH_Large_{i+1:02d}')
        cancer_types.append('BRCA')

    # Group 5: CON2-prominent (focal amplification, 8 samples)
    for i in range(8):
        weights = {'CON2': np.random.uniform(0.2, 0.5)}
        remaining = 1 - weights['CON2']
        weights['CON4'] = remaining * np.random.uniform(0.2, 0.4)
        weights['CON1'] = remaining * np.random.uniform(0.2, 0.3)
        weights['CON5'] = remaining * np.random.uniform(0.1, 0.3)
        weights['CON3'] = max(0, remaining - weights['CON4'] - weights['CON1'] - weights['CON5'])
        samples.append(generate_sample(weights))
        sample_names.append(f'Sample_FocalAmp_{i+1:02d}')
        cancer_types.append('PRAD' if i < 4 else 'CRC')

    # Create DataFrame
    df = pd.DataFrame(samples, columns=FEATURE_NAMES)
    df.insert(0, 'Sample_ID', sample_names)

    return df

if __name__ == '__main__':
    print("Generating realistic example dataset...")
    df = generate_example_dataset(n_samples=40)

    output_path = '/Users/adary/Documents/cna_sig/consensus/CON_fitting_app/examples/example_matrix.tsv'
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(df)} samples to {output_path}")

    # Show summary statistics
    print("\nSample groups:")
    print("- Diploid-like (CON5 dominant): 10 samples")
    print("- LOH + gains (CON1 dominant): 8 samples")
    print("- Mixed aneuploid (CON4 dominant): 8 samples")
    print("- Large LOH (CON3 dominant): 6 samples")
    print("- Focal amplification (CON2 prominent): 8 samples")
