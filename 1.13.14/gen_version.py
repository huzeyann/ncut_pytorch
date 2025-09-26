"""Generate version information for MkDocs."""
import tomllib
from pathlib import Path
import mkdocs_gen_files

# Read version from pyproject.toml
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

version = pyproject_data["project"]["version"]

# Generate a version file
with mkdocs_gen_files.open("version.md", "w") as f:
    f.write(f"# Version Information\n\n")
    f.write(f"Current version: **{version}**\n\n")
    f.write(f"This documentation was built for ncut_pytorch version {version}.\n")

# Update the site description with current version
site_config = {
    "site_description": f"Normalized Cut and Nyström Approximation - v{version}"
}

# You can also use this to inject version into templates
mkdocs_gen_files.set_edit_path("version.md", None)
