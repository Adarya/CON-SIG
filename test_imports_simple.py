#!/usr/bin/env python3
"""
Simple import test to check all dependencies
"""

print("Testing basic imports...")

try:
    import sys
    print(f"✅ Python: {sys.version}")
except ImportError as e:
    print(f"❌ Python import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import scipy
    print(f"✅ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"❌ SciPy import failed: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    import streamlit as st
    print(f"✅ Streamlit: {st.__version__}")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

print("\nTesting CON_fitting imports...")

try:
    from CON_fitting.src.signature_fitter import ConsensusSignatureFitter
    print("✅ CON_fitting.src.signature_fitter imported")
except ImportError as e:
    print(f"❌ CON_fitting.src.signature_fitter import failed: {e}")

try:
    from CON_fitting.src.data_processor import DataProcessor
    print("✅ CON_fitting.src.data_processor imported")
except ImportError as e:
    print(f"❌ CON_fitting.src.data_processor import failed: {e}")

try:
    from CON_fitting_enhancements.bootstrapped_signature_fitter import BootstrappedSignatureFitter
    print("✅ CON_fitting_enhancements.bootstrapped_signature_fitter imported")
except ImportError as e:
    print(f"❌ CON_fitting_enhancements.bootstrapped_signature_fitter import failed: {e}")

print("\nTesting backend imports...")

try:
    from backend import load_consensus_signatures, run_signature_fitting
    print("✅ Backend functions imported")
except ImportError as e:
    print(f"❌ Backend import failed: {e}")

print("\nAll import tests completed!")