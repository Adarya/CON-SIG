#!/usr/bin/env python3
"""
Test the full backend functionality
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_full_signature_fitting():
    """Test the full signature fitting workflow"""
    print("Testing full signature fitting workflow...")
    
    try:
        from backend import run_signature_fitting, get_example_data
        
        # Get example data
        examples = get_example_data()
        example_matrix = examples['example_matrix']
        
        print(f"Example matrix shape: {example_matrix.shape}")
        print(f"Example matrix columns: {list(example_matrix.columns)}")
        
        # Test regular fitting
        print("\n--- Testing regular fitting ---")
        results = run_signature_fitting(
            example_matrix,
            use_bootstrap=False,
            method='nnls'
        )
        
        print(f"✅ Regular fitting completed!")
        print(f"   - Activities shape: {results['activities'].shape}")
        print(f"   - Mean R²: {results['mean_r2']:.3f}")
        print(f"   - Mean error: {results['mean_error']:.6f}")
        print(f"   - Method: {results['method']}")
        
        # Test bootstrap fitting (with fewer iterations for speed)
        print("\n--- Testing bootstrap fitting ---")
        bootstrap_results = run_signature_fitting(
            example_matrix,
            use_bootstrap=True,
            bootstrap_iterations=10,
            method='nnls'
        )
        
        print(f"✅ Bootstrap fitting completed!")
        print(f"   - Activities shape: {bootstrap_results['activities'].shape}")
        print(f"   - Mean R²: {bootstrap_results['mean_r2']:.3f}")
        print(f"   - Bootstrap iterations: {bootstrap_results['bootstrap_iterations']}")
        print(f"   - Has confidence intervals: {'confidence_intervals' in bootstrap_results}")
        
        if 'confidence_intervals' in bootstrap_results:
            ci = bootstrap_results['confidence_intervals']
            print(f"   - CI lower shape: {ci['lower'].shape}")
            print(f"   - CI upper shape: {ci['upper'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full signature fitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_elastic_net_method():
    """Test the elastic net method"""
    print("\nTesting elastic net method...")
    
    try:
        from backend import run_signature_fitting, get_example_data
        
        # Get example data
        examples = get_example_data()
        example_matrix = examples['example_matrix']
        
        # Test elastic net fitting
        results = run_signature_fitting(
            example_matrix,
            use_bootstrap=False,
            method='elastic_net'
        )
        
        print(f"✅ Elastic net fitting completed!")
        print(f"   - Activities shape: {results['activities'].shape}")
        print(f"   - Mean R²: {results['mean_r2']:.3f}")
        print(f"   - Method: {results['method']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Elastic net test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧬 Full Backend Test Suite")
    print("=" * 40)
    
    tests = [
        test_full_signature_fitting,
        test_elastic_net_method
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✅ All backend functionality working!")
        return True
    else:
        print("❌ Some backend tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)