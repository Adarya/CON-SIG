#!/usr/bin/env python3
"""
Dependency fix script for CONSIG
"""

import sys
import subprocess
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔧 CONSIG Dependency Fix Script")
    print("=" * 40)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Install/upgrade required packages
    packages = [
        "scipy>=1.9.0",
        "numpy>=1.21.0", 
        "pandas>=1.5.0",
        "streamlit>=1.35.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    print("\n📦 Installing/upgrading packages...")
    for package in packages:
        print(f"Installing {package}...")
        success, stdout, stderr = run_command(f"pip install '{package}'")
        if success:
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}: {stderr}")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        ("scipy", "import scipy"),
        ("numpy", "import numpy"),
        ("pandas", "import pandas"),
        ("streamlit", "import streamlit"),
        ("sklearn", "import sklearn"),
        ("matplotlib", "import matplotlib"),
        ("CON_fitting", "from CON_fitting.src.signature_fitter import ConsensusSignatureFitter")
    ]
    
    all_good = True
    for name, import_cmd in test_imports:
        success, _, stderr = run_command(f"python -c \"{import_cmd}\"")
        if success:
            print(f"✅ {name}")
        else:
            print(f"❌ {name}: {stderr}")
            all_good = False
    
    if all_good:
        print("\n🎉 All dependencies are working correctly!")
        print("You can now run: streamlit run app.py")
    else:
        print("\n⚠️ Some imports failed. Try:")
        print("1. Restart your terminal/command prompt")
        print("2. Run this script again")
        print("3. Check your Python environment")

if __name__ == "__main__":
    main()