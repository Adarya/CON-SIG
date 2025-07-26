#!/usr/bin/env python3
"""
Direct Streamlit launch script to test imports
"""

import sys
import os
from pathlib import Path

print("🧬 CONSIG - Direct Launch Test")
print("=" * 40)
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")

# Test critical imports
try:
    import scipy
    print(f"✅ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"❌ SciPy import failed: {e}")
    sys.exit(1)

try:
    import streamlit as st
    print(f"✅ Streamlit: {st.__version__}")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")
    sys.exit(1)

try:
    from CON_fitting.src.signature_fitter import ConsensusSignatureFitter
    print("✅ CON_fitting modules accessible")
except ImportError as e:
    print(f"❌ CON_fitting import failed: {e}")
    sys.exit(1)

print("✅ All imports successful!")
print("🚀 Launching Streamlit app...")

# Launch streamlit
import subprocess
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])