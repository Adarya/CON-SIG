# CON_fitting Examples

This directory contains example scripts demonstrating how to use the CON_fitting framework.

## simple_example.py

A comprehensive example script that can work with both synthetic and real CNA data.

### Features

- **Automatic Data Orientation Detection**: Intelligently detects whether samples are in rows or columns
- **Multiple Input Formats**: Supports CSV and TSV files
- **Manual Override Options**: Force transpose behavior when needed
- **Comprehensive Output**: Generates fitted activities, quality metrics, and visualizations

### Usage

#### 1. Synthetic Data (Default)
```bash
python simple_example.py
```
Creates synthetic CNA data and fits consensus signatures for demonstration.

#### 2. Real Data with Auto-Detection
```bash
python simple_example.py your_data.csv
python simple_example.py your_data.tsv
```
Automatically detects data orientation by analyzing overlap with consensus signature categories.

#### 3. Manual Transpose Control
```bash
# Force transpose (when samples are columns)
python simple_example.py your_data.csv --transpose

# Force no transpose (when samples are rows)  
python simple_example.py your_data.csv --no-transpose
```

### Data Format Requirements

#### Expected CNA Categories
Your data should contain these CNA categories (matching consensus signatures):
- `0:homdel:0-100kb`, `0:homdel:100kb-1Mb`, `0:homdel:>1Mb`
- `1:LOH:0-100kb`, `1:LOH:100kb-1Mb`, `1:LOH:1Mb-10Mb`, `1:LOH:10Mb-40Mb`, `1:LOH:>40Mb`
- `2:het:0-100kb`, `2:het:100kb-1Mb`, `2:het:1Mb-10Mb`, `2:het:10Mb-40Mb`, `2:het:>40Mb`
- `3-4:het:0-100kb`, `3-4:het:100kb-1Mb`, `3-4:het:1Mb-10Mb`, `3-4:het:10Mb-40Mb`, `3-4:het:>40Mb`
- `5-8:het:0-100kb`, `5-8:het:100kb-1Mb`, `5-8:het:1Mb-10Mb`, `5-8:het:10Mb-40Mb`, `5-8:het:>40Mb`
- `9+:het:0-100kb`, `9+:het:100kb-1Mb`, `9+:het:1Mb-10Mb`, `9+:het:10Mb-40Mb`, `9+:het:>40Mb`

#### Samples as Rows Format (Preferred)
```csv
SampleName,0:homdel:0-100kb,0:homdel:100kb-1Mb,...
Sample1,0.1,0.05,...
Sample2,0.2,0.08,...
```

#### Samples as Columns Format (Common)
```csv
MutationType,Sample1,Sample2,Sample3,...
0:homdel:0-100kb,0.1,0.2,0.15,...
0:homdel:100kb-1Mb,0.05,0.08,0.06,...
```

### Auto-Detection Logic

The script analyzes your data to determine orientation:

1. **Checks columns**: How many match consensus CNA categories?
2. **Checks index**: How many match consensus CNA categories?
3. **Chooses best match**: Higher overlap percentage wins
4. **Transposes if needed**: If index has better overlap, transpose

Example output:
```
Data orientation analysis:
- Columns as CNA categories: 28/28 match (100.0%)
- Index as CNA categories: 0/3 match (0.0%)
→ Detected: Samples are ROWS (no transpose needed)
```

### Output Files

The script generates:

- **`fitted_activities.csv`** - Signature activities for each sample
- **`processed_cna_data.csv`** - Preprocessed input data
- **`analysis_summary.csv`** - Analysis metadata and quality metrics
- **`fitted_activities.png`** - Heatmap visualization
- **`quality_metrics.png`** - Quality assessment plots

Output location depends on input:
- Synthetic data: `../output/example/`
- Real data: `../output/real_data_{filename}/`

### Test Files Included

- **`test_data_samples_as_columns.csv`** - Example with samples as columns
- **`test_data_samples_as_rows.csv`** - Example with samples as rows

Try these:
```bash
python simple_example.py test_data_samples_as_columns.csv
python simple_example.py test_data_samples_as_rows.csv
```

### Troubleshooting

#### Common Issues

1. **"No common CNA categories found"**
   - Check that your CNA category names exactly match the consensus signatures
   - Verify file format and delimiter (CSV vs TSV)

2. **Poor fitting quality (low R²)**
   - Check data quality and preprocessing
   - Ensure CNA values are non-negative
   - Try different fitting methods in the full framework

3. **Wrong orientation detected**
   - Use `--transpose` or `--no-transpose` to override auto-detection
   - Check that your CNA categories match expected names

#### Error Messages

The script provides helpful error messages and troubleshooting tips:
```
Error: [Error description]

Troubleshooting tips:
1. Check that your file exists and is readable
2. Ensure the file is in CSV or TSV format  
3. Try using --transpose if samples are in columns
4. Check that CNA category names match the consensus signatures
```

### Integration with Full Framework

This example demonstrates basic usage. For production analysis:

1. Use the full `test_validation.py` for comprehensive validation
2. Leverage all modules in `src/` for advanced functionality
3. Customize fitting parameters and methods as needed
4. Implement batch processing for large datasets

## Next Steps

- Run the example with your own data
- Explore the full framework capabilities
- Review the validation results and quality metrics
- Integrate into your analysis pipeline 