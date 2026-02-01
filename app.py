"""
CONSIG - CON_fitting Streamlit Web Application

A user-friendly web interface for CNA signature analysis using the CON_fitting framework.
Allows users to upload CNA segment files or pre-processed matrices and get signature activities
with optional bootstrap uncertainty estimation.
"""

import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from typing import Optional, Tuple, Dict, Any

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our backend functions
from backend import (
    load_user_file,
    run_signature_fitting,
    get_file_statistics,
    validate_file_format
)
from plotting import create_stacked_bar_plot, save_plot_as_bytes

# Configure page
st.set_page_config(
    page_title="CONSIG - CNA Signature Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cloud deployment detection and optimization
import os
IS_CLOUD = any(indicator in str(os.environ.get('PATH', '')) 
               for indicator in ['streamlit', 'share.streamlit.io'])

if IS_CLOUD:
    # Reduce default bootstrap iterations for cloud deployment
    DEFAULT_BOOTSTRAP_ITERATIONS = 50
    MAX_BOOTSTRAP_ITERATIONS = 100
    st.sidebar.info("🌐 Running on Streamlit Cloud - Bootstrap iterations limited for performance")
else:
    DEFAULT_BOOTSTRAP_ITERATIONS = 200
    MAX_BOOTSTRAP_ITERATIONS = 1000

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .download-section {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'file_data' not in st.session_state:
        st.session_state.file_data = None
    if 'file_stats' not in st.session_state:
        st.session_state.file_stats = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = None

def load_demo_data():
    """Load the example dataset for demonstration."""
    demo_path = Path(__file__).parent / 'examples' / 'example_matrix.tsv'
    if demo_path.exists():
        return pd.read_csv(demo_path, sep='\t')
    return None

def create_download_link(data: bytes, filename: str, file_type: str) -> str:
    """Create a download link for files"""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {file_type}</a>'

def display_file_info(stats: Dict[str, Any]):
    """Display file information in metrics format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Samples", stats.get('n_samples', 'N/A'))
    with col2:
        st.metric("Features", stats.get('n_features', 'N/A'))
    with col3:
        st.metric("File Type", stats.get('file_type', 'N/A'))

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🧬 CONSIG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CNA Signature Analysis Web Application</div>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Analysis Parameters")
        
        # File upload
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload CNA data file",
            type=['seg', 'tsv', 'csv'],
            help="Upload either a cbioportal .seg file or a pre-processed CNA matrix (.tsv/.csv)"
        )

        # Demo data button
        st.markdown("---")
        if st.button("📊 Load Demo Data", help="Load a pre-built example dataset with 40 samples representing diverse CNA profiles"):
            st.session_state.demo_mode = True
            st.session_state.demo_data = load_demo_data()
            st.session_state.results = None  # Reset results
            st.rerun()

        if st.session_state.demo_mode:
            st.success("✅ Demo data loaded (40 samples)")
            if st.button("❌ Clear Demo Data"):
                st.session_state.demo_mode = False
                st.session_state.demo_data = None
                st.session_state.results = None
                st.rerun()
        
        # Analysis parameters
        st.subheader("🔬 Analysis Options")
        
        use_bootstrap = st.checkbox(
            "Use bootstrap uncertainty estimation",
            value=False,
            help="Enable bootstrap resampling for confidence intervals (slower but more robust)"
        )
        
        bootstrap_iterations = st.number_input(
            "Bootstrap iterations",
            min_value=50,
            max_value=MAX_BOOTSTRAP_ITERATIONS,
            value=DEFAULT_BOOTSTRAP_ITERATIONS,
            step=50,
            disabled=not use_bootstrap,
            help="Number of bootstrap iterations (higher = more accurate but slower)"
        )
        
        deconv_method = st.selectbox(
            "Deconvolution method",
            options=['nnls', 'elastic_net'],
            index=0,
            help="Method for signature deconvolution"
        )
        
        # Analysis button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Info section
        st.subheader("ℹ️ About")
        st.markdown("""
        **CONSIG** provides web-based access to CNA signature analysis.
        
        **Supported formats:**
        - Raw cbioportal .seg files
        - Pre-processed CNA matrices (.tsv/.csv)
        
        **Features:**
        - Signature activity estimation
        - Bootstrap uncertainty quantification
        - Interactive visualizations
        - Download results (CSV, PNG, PDF)
        """)
    
    # Main content area
    # Check if we have data (either uploaded or demo)
    has_data = uploaded_file is not None or st.session_state.demo_mode

    if has_data:
        # Display file information
        st.subheader("📄 File Information")

        try:
            # Handle demo mode vs uploaded file
            if st.session_state.demo_mode and st.session_state.demo_data is not None:
                demo_df = st.session_state.demo_data
                file_stats = {
                    'n_samples': len(demo_df),
                    'n_features': len(demo_df.columns) - 1,  # Exclude Sample_ID
                    'file_type': 'demo'
                }
                st.info("📊 **Demo Mode**: Using example dataset with 40 samples representing diverse CNA profiles (diploid-like, LOH-dominant, mixed aneuploid, focal amplification patterns)")
            else:
                # Validate and get file statistics
                file_stats = get_file_statistics(uploaded_file)

            st.session_state.file_stats = file_stats
            display_file_info(file_stats)
            
            # Show file preview
            with st.expander("📊 Data Preview"):
                if st.session_state.demo_mode and st.session_state.demo_data is not None:
                    st.write("**Demo dataset: 40 samples with diverse CNA patterns**")
                    st.dataframe(st.session_state.demo_data.head(5), use_container_width=True)
                elif file_stats['file_type'] == 'seg':
                    st.write("**cbioportal segment file format detected**")
                    # Show first few rows of seg file
                    preview_df = pd.read_csv(uploaded_file, sep='\t', nrows=5)
                    st.dataframe(preview_df, use_container_width=True)
                else:
                    st.write("**Pre-processed CNA matrix detected**")
                    # Show first few rows of matrix
                    preview_df = pd.read_csv(uploaded_file, sep='\t' if file_stats['file_type'] == 'tsv' else ',', nrows=5)
                    st.dataframe(preview_df, use_container_width=True)
            
            # Analysis section
            if run_analysis:
                st.subheader("🔬 Analysis Results")
                
                with st.spinner("Processing data and fitting signatures..."):
                    try:
                        # Load and process file (demo or uploaded)
                        if st.session_state.demo_mode and st.session_state.demo_data is not None:
                            # Use demo data directly
                            demo_df = st.session_state.demo_data.copy()
                            demo_df = demo_df.set_index('Sample_ID')
                            matrix_data = demo_df
                        else:
                            matrix_data = load_user_file(uploaded_file)
                        st.session_state.file_data = matrix_data
                        
                        # Run signature fitting
                        results = run_signature_fitting(
                            matrix_data,
                            use_bootstrap=use_bootstrap,
                            bootstrap_iterations=bootstrap_iterations,
                            method=deconv_method
                        )
                        
                        st.session_state.results = results
                        
                        st.success("✅ Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)
                        return
            
            # Display results if available
            if st.session_state.results is not None:
                results = st.session_state.results
                
                # Quality metrics
                st.subheader("📊 Quality Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mean R²", f"{results['mean_r2']:.3f}")
                with col2:
                    st.metric("Mean Reconstruction Error", f"{results['mean_error']:.6f}")
                
                # Results tabs
                tab1, tab2 = st.tabs(["📈 Signature Activities", "📊 Visualization"])
                
                with tab1:
                    st.subheader("Signature Activities")
                    
                    # Display activities table
                    activities_df = results['activities']
                    st.dataframe(activities_df, use_container_width=True)
                    
                    # Download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.write("**Download Results:**")
                    
                    # CSV download
                    csv_buffer = io.StringIO()
                    activities_df.to_csv(csv_buffer, index=True)
                    csv_data = csv_buffer.getvalue().encode()
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="signature_activities.csv",
                        mime="text/csv"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("Signature Contributions")
                    
                    with st.spinner("Generating visualization..."):
                        try:
                            # Create stacked bar plot
                            fig = create_stacked_bar_plot(results['activities'])
                            st.pyplot(fig)
                            
                            # Download buttons for plots
                            st.markdown('<div class="download-section">', unsafe_allow_html=True)
                            st.write("**Download Plot:**")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # PNG download
                                png_data = save_plot_as_bytes(fig, format='png')
                                st.download_button(
                                    label="📥 Download PNG",
                                    data=png_data,
                                    file_name="signature_plot.png",
                                    mime="image/png"
                                )
                            
                            with col2:
                                # PDF download
                                pdf_data = save_plot_as_bytes(fig, format='pdf')
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=pdf_data,
                                    file_name="signature_plot.pdf",
                                    mime="application/pdf"
                                )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error generating plot: {str(e)}")
                            st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to CONSIG! 🎯

        To get started:
        1. **Upload your CNA data file** using the sidebar, **OR click "Load Demo Data"** to try with example data
        2. **Configure analysis parameters** (bootstrap, method, etc.)
        3. **Click "Run Analysis"** to process your data
        4. **View and download results** in the main area

        **Supported file formats:**
        - **cbioportal .seg files**: Raw segment data with columns: ID, chrom, loc.start, loc.end, seg.mean
        - **Pre-processed matrices**: .tsv/.csv files with samples as rows and CNA features as columns

        **Try the Demo:** Click "Load Demo Data" in the sidebar to analyze 40 example samples with diverse CNA patterns including diploid-like, LOH-dominant, mixed aneuploid, and focal amplification profiles.

        **Need help?** Check the sidebar for more information about the analysis options.
        """)

if __name__ == "__main__":
    main()