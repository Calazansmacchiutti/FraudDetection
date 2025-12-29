"""
Setup script for KYC/KYT Fraud Detection package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="kyc-kyt-fraud-detection",
    version="1.0.0",
    description="Loan Default Prediction using KYC/KYT approach with Optuna optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/kyc-kyt-fraud-detection",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="fraud detection, loan default, KYC, KYT, machine learning, optuna",
    entry_points={
        "console_scripts": [
            "kyc-train=scripts.train:main",
        ],
    },
)
