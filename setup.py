"""
Setup script cho Video Fraud Detection System
"""

from setuptools import setup, find_packages
import sys
import os

# Đọc README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Đọc requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="video-fraud-detection",
    version="1.0.0",
    author="Anh Vũ",
    author_email="your.email@example.com",
    description="Hệ thống phát hiện gian lận trong video sử dụng AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-fraud-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "web": [
            "Flask>=2.2.0",
            "Werkzeug>=2.2.0",
        ],
        "performance": [
            "accelerate",
            "xformers",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-fraud-detection=run_analyzer:main",
            "fraud-detection-gui=run_analyzer:main",
            "fraud-detection-web=app:app",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="video analysis, fraud detection, AI, computer vision, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/video-fraud-detection/issues",
        "Source": "https://github.com/yourusername/video-fraud-detection",
        "Documentation": "https://github.com/yourusername/video-fraud-detection/wiki",
    },
)
