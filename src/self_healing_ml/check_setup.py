import sys
import joblib
import numpy as np
import pandas as pd
import sklearn

def main():
    print("Self-Healing ML Platform")
    print("-" * 35)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"NumPy version     : {np.__version__}")
    print(f"Pandas version    : {pd.__version__}")
    print(f"Scikit-learn      : {sklearn.__version__}")
    print(f"Joblib version    : {joblib.__version__}")
    print("\nEnvironment verification successful!")

if __name__ == "__main__":
    main()
