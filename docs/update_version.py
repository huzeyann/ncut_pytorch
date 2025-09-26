#!/usr/bin/env python3
"""
Script to update MkDocs site description with current version.
This can be run as part of the RTD build process.
"""
import os
import sys
from pathlib import Path

# Try to import tomllib (Python 3.11+) or fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: Neither tomllib nor tomli is available. Install tomli: pip install tomli")
        sys.exit(1)

def get_version():
    """Get version from pyproject.toml or environment variable."""
    # First try to get version from RTD environment variable
    rtd_version = os.environ.get('READTHEDOCS_VERSION')
    if rtd_version and rtd_version not in ['latest', 'stable']:
        return rtd_version
    
    # Fallback to reading from pyproject.toml
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        return pyproject_data["project"]["version"]
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
        return "unknown"

def main():
    version = get_version()
    site_description = f"Normalized Cut and Nyström Approximation - v{version}"
    
    # Set environment variable for MkDocs
    os.environ['SITE_DESCRIPTION'] = site_description
    
    print(f"Set SITE_DESCRIPTION to: {site_description}")
    return site_description

if __name__ == "__main__":
    main()
