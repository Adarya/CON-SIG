#!/usr/bin/env python3
"""
Complete workflow test for CONSIG application
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_seg_file_processing():
    """Test .seg file processing workflow"""
    print("Testing .seg file processing...")
    
    try:
        # Create a mock seg file content
        seg_content = """ID	chrom	loc.start	loc.end	seg.mean
Test_Sample_1	1	1000000	5000000	0.2
Test_Sample_1	1	5000001	10000000	-0.3
Test_Sample_1	2	1000000	8000000	0.1
Test_Sample_2	1	1000000	4000000	0.3
Test_Sample_2	1	4000001	9000000	-0.1
Test_Sample_2	2	1000000	7000000	0.2"""
        
        # Create mock uploaded file
        class MockUploadedFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content.encode()
                self.position = 0
                
            def read(self):
                return self.content
                
            def seek(self, position):
                self.position = position
                
            def getvalue(self):
                return self.content
        
        mock_seg_file = MockUploadedFile("test.seg", seg_content)
        
        # Test file statistics
        from backend import get_file_statistics
        stats = get_file_statistics(mock_seg_file)
        print(f"✅ .seg file statistics: {stats}")
        
        # Test file processing
        from backend import load_user_file
        matrix = load_user_file(mock_seg_file)
        print(f"✅ .seg file processed to matrix: {matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ .seg file processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matrix_file_processing():
    """Test matrix file processing workflow"""
    print("\nTesting matrix file processing...")
    
    try:
        # Create a mock matrix file content
        matrix_content = """Sample_ID	0:homdel:0-100kb	0:homdel:100kb-1Mb	0:homdel:>1Mb	1:LOH:0-100kb	1:LOH:100kb-1Mb	1:LOH:1Mb-10Mb	1:LOH:10Mb-40Mb	1:LOH:>40Mb	2:het:0-100kb	2:het:100kb-1Mb	2:het:1Mb-10Mb	2:het:10Mb-40Mb	2:het:>40Mb	3-4:het:0-100kb	3-4:het:100kb-1Mb	3-4:het:1Mb-10Mb	3-4:het:10Mb-40Mb	3-4:het:>40Mb	5-8:het:0-100kb	5-8:het:100kb-1Mb	5-8:het:1Mb-10Mb	5-8:het:10Mb-40Mb	5-8:het:>40Mb	9+:het:0-100kb	9+:het:100kb-1Mb	9+:het:1Mb-10Mb	9+:het:10Mb-40Mb	9+:het:>40Mb
Test_Sample_1	0	1	0	2	3	1	0	1	5	4	3	2	1	1	2	1	0	1	0	1	0	1	0	0	1	0	0	1
Test_Sample_2	1	0	1	1	2	2	1	0	4	3	4	3	2	2	1	2	1	0	1	0	1	0	1	1	0	1	0	0"""
        
        # Create mock uploaded file
        class MockUploadedFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content.encode()
                self.position = 0
                
            def read(self):
                return self.content
                
            def seek(self, position):
                self.position = position
                
            def getvalue(self):
                return self.content
        
        mock_matrix_file = MockUploadedFile("test.tsv", matrix_content)
        
        # Test file statistics
        from backend import get_file_statistics
        stats = get_file_statistics(mock_matrix_file)
        print(f"✅ Matrix file statistics: {stats}")
        
        # Test file processing
        from backend import load_user_file
        matrix = load_user_file(mock_matrix_file)
        print(f"✅ Matrix file processed: {matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix file processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_analysis_workflow():
    """Test complete analysis workflow"""
    print("\nTesting complete analysis workflow...")
    
    try:
        # Get example data
        from backend import get_example_data, run_signature_fitting
        
        examples = get_example_data()
        example_matrix = examples['example_matrix']
        
        print(f"Using example matrix: {example_matrix.shape}")
        
        # Test regular analysis
        results = run_signature_fitting(
            example_matrix,
            use_bootstrap=False,
            method='nnls'
        )
        
        print(f"✅ Regular analysis completed:")
        print(f"   - Activities: {results['activities'].shape}")
        print(f"   - Mean R²: {results['mean_r2']:.3f}")
        print(f"   - Signatures: {list(results['activities'].columns)}")
        
        # Test bootstrap analysis
        bootstrap_results = run_signature_fitting(
            example_matrix,
            use_bootstrap=True,
            bootstrap_iterations=5,  # Small number for testing
            method='nnls'
        )
        
        print(f"✅ Bootstrap analysis completed:")
        print(f"   - Activities: {bootstrap_results['activities'].shape}")
        print(f"   - Mean R²: {bootstrap_results['mean_r2']:.3f}")
        print(f"   - Has confidence intervals: {'confidence_intervals' in bootstrap_results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete analysis workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plotting_functionality():
    """Test plotting functionality"""
    print("\nTesting plotting functionality...")
    
    try:
        from backend import get_example_data, run_signature_fitting
        from plotting import create_stacked_bar_plot, save_plot_as_bytes
        
        # Get example data and run analysis
        examples = get_example_data()
        example_matrix = examples['example_matrix']
        
        results = run_signature_fitting(
            example_matrix,
            use_bootstrap=False,
            method='nnls'
        )
        
        # Create plot
        fig = create_stacked_bar_plot(results['activities'])
        print(f"✅ Stacked bar plot created: {type(fig)}")
        
        # Test plot saving
        png_data = save_plot_as_bytes(fig, format='png')
        print(f"✅ Plot saved as PNG: {len(png_data)} bytes")
        
        pdf_data = save_plot_as_bytes(fig, format='pdf')
        print(f"✅ Plot saved as PDF: {len(pdf_data)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Plotting functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all workflow tests"""
    print("🧬 CONSIG Complete Workflow Test Suite")
    print("=" * 50)
    
    tests = [
        test_seg_file_processing,
        test_matrix_file_processing,
        test_complete_analysis_workflow,
        test_plotting_functionality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✅ Complete workflow working perfectly!")
        print("\n🎉 CONSIG is ready for use!")
        print("   Start the application with: ./run_app.sh")
        return True
    else:
        print("❌ Some workflow tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)