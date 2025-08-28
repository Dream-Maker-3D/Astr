#!/usr/bin/env python3
"""
Test runner for Astir Voice Assistant.
Properly sets up the Python path and runs tests.
"""

import sys
import os
import subprocess

def setup_python_path():
    """Add src directory to Python path."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if src_path not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{src_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = src_path

def run_tests():
    """Run the test suite."""
    setup_python_path()
    
    # Run pytest with proper configuration
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '-v',
        '--tb=short',
        '--no-header'
    ]
    
    print("🧪 Running Astir Voice Assistant Tests")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
