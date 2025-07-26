#!/usr/bin/env python3
"""
Test script to verify backend functionality
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

def test_file_statistics():
    """Test file statistics function"""
    print("Testing file statistics...")
    
    # Create a mock uploaded file object
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
            self.position = 0
            
        def read(self):
            return self.content
            
        def seek(self, position):
            self.position = position
            
        def getvalue(self):
            return self.content.encode() if isinstance(self.content, str) else self.content
    
    # Test with TSV matrix
    tsv_content = """Sample_ID	Feature_1	Feature_2	Feature_3
Sample_001	0.1	0.2	0.3
Sample_002	0.4	0.5	0.6"""
    
    mock_file = MockUploadedFile("test.tsv", tsv_content)
    
    try:
        from backend import get_file_statistics
        stats = get_file_statistics(mock_file)
        
        print(f"✅ File statistics test passed: {stats}")
        return True
        
    except Exception as e:
        print(f"❌ File statistics test failed: {e}")
        return False

def test_matrix_validation():
    """Test matrix validation"""
    print("\nTesting matrix validation...")
    
    # Create test matrix
    test_matrix = pd.DataFrame({
        'Feature_1': [0.1, 0.2, 0.3],
        'Feature_2': [0.4, 0.5, 0.6],
        'Feature_3': [0.7, 0.8, 0.9]
    }, index=['Sample_1', 'Sample_2', 'Sample_3'])
    
    try:
        from backend import validate_cna_matrix
        is_valid, message = validate_cna_matrix(test_matrix)
        
        if is_valid:
            print(f"✅ Matrix validation test passed: {message}")
            return True
        else:
            print(f"❌ Matrix validation test failed: {message}")
            return False
            
    except Exception as e:
        print(f"❌ Matrix validation test failed: {e}")
        return False

def test_example_data():
    """Test example data generation"""
    print("\nTesting example data generation...")
    
    try:
        from backend import get_example_data
        examples = get_example_data()
        
        if 'example_matrix' in examples:
            matrix = examples['example_matrix']
            print(f"✅ Example data test passed: {matrix.shape}")
            return True
        else:
            print("❌ No example matrix found")
            return False
            
    except Exception as e:
        print(f"❌ Example data test failed: {e}")
        return False

def test_imports():
    """Test that required modules can be imported"""
    print("\nTesting imports...")
    
    try:
        from backend import (
            validate_file_format,
            get_file_statistics,
            validate_cna_matrix,
            get_example_data
        )
        print("✅ All backend functions imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧬 CONSIG Backend Test Suite")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_file_statistics,
        test_matrix_validation,
        test_example_data
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 30)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)