from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'streamlit==0.65.2',
    "pip>=9",
    "setuptools>=26",
    "wheel>=0.29",
    "pandas==2.9.0",
    "pytest==6.1.2",
    "coverage==5.3",
    #"flake8",
    #"black",
    #"yapf",
    #"python-gitlab",
    #"twine",
    "flask==1.1.1",
    "streamlit==0.71.0",
    "scikit-learn==0.23.2",
    "numpy==1.18.5"]

setup(
    name='StreamlitApp',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Streamlit App'
)
