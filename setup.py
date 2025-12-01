"""Setup script for Cat Chess."""

from setuptools import setup, find_packages

setup(
    name="cat-chess",
    version="1.0.0",
    description="A three-faction chess variant with peacekeeping cats",
    author="Cat Chess Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0"],
    },
    entry_points={
        "console_scripts": [
            "cat-chess=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
