"""Generate version information for MkDocs."""
import os
from pathlib import Path
import mkdocs_gen_files

# Try to import tomllib (Python 3.11+) or fallback to tomli
try:
    import tomllib
except ImportError:
    import tomli as tomllib

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

version = get_version()

# Generate a version file
with mkdocs_gen_files.open("version.md", "w") as f:
    f.write(f"# Version Information\n\n")
    f.write(f"Current version: **{version}**\n\n")
    f.write(f"This documentation was built for ncut_pytorch version {version}.\n\n")
    
    # Add RTD-specific information if available
    rtd_version = os.environ.get('READTHEDOCS_VERSION')
    if rtd_version:
        f.write(f"Read the Docs version: `{rtd_version}`\n\n")
    
    build_id = os.environ.get('READTHEDOCS_VERSION_NAME')
    if build_id:
        f.write(f"Build ID: `{build_id}`\n\n")

# You can also use this to inject version into templates
mkdocs_gen_files.set_edit_path("version.md", None)
