#!/usr/bin/env python3

"""
Quick Start Script for Scaling Study
Runs a smaller-scale version of the study for testing and quick results.
"""

import subprocess
import sys

def main():
    print("🚀 Starting Scaling Study - Quick Version")
    print("This will train models on 3 different problem sizes and evaluate them.")
    print("Training time per size: ~10 minutes (1M timesteps)")
    print()
    
    # Run scaling study with reduced parameters for quick testing
    cmd = [
        "python", "scaling_study.py",
        "--mode", "full",
        "--study-name", "quick_scaling_test",
        "--timesteps-per-size", "1000000",  # Reduced from 3M to 1M
        "--eval-episodes", "20",           # Reduced from 50 to 20
        "--sizes", "15x15_small", "20x20_medium", "50x15_large",  # Just 3 sizes
        "--include-heuristics"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Quick scaling study completed successfully!")
        print("📁 Check the scaling_studies/ directory for results")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Study failed with return code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Study interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 