"""
Setup script for Research Agent
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="research-agent2",
    version="1.0.0",
    description="An intelligent assistant for conducting thorough research and analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Research Agent Team",
    author_email="contact@research-agent.com",
    url="https://github.com/tkim/research_agent2",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'research_agent2': ['config.json'],
    },
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-asyncio>=0.18.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.910',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'research-agent2=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    python_requires=">=3.7",
    keywords="research, web-search, citations, analysis, academic, information-retrieval",
    project_urls={
        "Bug Reports": "https://github.com/tkim/research_agent2/issues",
        "Source": "https://github.com/tkim/research_agent2",
        "Documentation": "https://github.com/tkim/research_agent2/wiki",
    },
)