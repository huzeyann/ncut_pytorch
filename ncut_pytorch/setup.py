from setuptools import setup, find_packages


# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# read the requirements file
import os
requirement_file = 'requirements.txt'
with open(requirement_file) as f:
    required = f.read().splitlines()
    
setup(
    name="ncut_pytorch",
    version="1.7.6",
    packages=['ncut_pytorch'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Huzheng Yang',
    author_email='huze.yann@gmail.com',
    install_requires=required,
)