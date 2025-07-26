#!/usr/bin/env python3
"""
Quick test script to verify all imports work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported"""
    
    print("Testing imports...")
    
    # Test standard library imports
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Standard libraries imported successfully")
    except ImportError as e:
        print(f"❌ Standard library import failed: {e}")
        return False
    
    # Test Streamlit (might not be installed yet)
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"⚠️  Streamlit not available: {e}")
        print("   This is expected if requirements haven't been installed yet")
    
    # Test our modules
    try:
        from backend import validate_file_format, get_example_data
        from plotting import create_stacked_bar_plot
        print("✅ Local modules imported successfully")
    except ImportError as e:
        print(f"❌ Local module import failed: {e}")
        return False
    
    # Test CON_fitting imports (might fail if not properly set up)
    try:
        from CON_fitting.src.data_processor import DataProcessor
        from CON_fitting.src.signature_fitter import ConsensusSignatureFitter
        print("✅ CON_fitting modules imported successfully")
    except ImportError as e:
        print(f"⚠️  CON_fitting modules not available: {e}")
        print("   This is expected if CON_fitting is not properly set up")
    
    return True

def test_example_data():
    """Test that example data can be loaded"""
    
    print("\nTesting example data...")
    
    try:
        from backend import get_example_data
        examples = get_example_data()
        
        if 'example_matrix' in examples:
            matrix = examples['example_matrix']
            print(f"✅ Example matrix generated: {matrix.shape}")
            print(f"   - Samples: {len(matrix)}")
            print(f"   - Features: {len(matrix.columns)}")
        else:
            print("❌ No example matrix found")
            return False
            
    except Exception as e:
        print(f"❌ Example data test failed: {e}")
        return False
    
    return True

def test_plotting():
    """Test that plotting functions work"""
    
    print("\nTesting plotting functions...")
    
    try:
        from backend import get_example_data
        from plotting import create_stacked_bar_plot
        
        # Generate example data
        examples = get_example_data()
        matrix = examples['example_matrix']
        
        # Create a simple plot
        fig = create_stacked_bar_plot(matrix, title="Test Plot")
        
        if fig is not None:
            print("✅ Plotting functions work correctly")
            return True
        else:
            print("❌ Plotting function returned None")
            return False
            
    except Exception as e:
        print(f"❌ Plotting test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧬 CONSIG Import Test Suite")
    print("=" * 30)
    
    success = True
    
    success &= test_imports()
    success &= test_example_data()
    success &= test_plotting()
    
    print("\n" + "=" * 30)
    if success:
        print("✅ All tests passed! The application should work correctly.")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1)