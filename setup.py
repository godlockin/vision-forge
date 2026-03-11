from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vision-forge",
    version="0.1.0",
    author="Vision Forge Team",
    description="Intelligent Vision Reviewer - Expert system for AI image evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.5.0",
        "pyyaml>=6.0.1",
        "click>=8.1.7",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.2.1",
        "aiohttp>=3.9.1",
        "openai>=1.3.0",
        "google-cloud-aiplatform>=1.38.0",
        "vertexai>=1.38.0",
        "httpx>=0.25.0",
        "Pillow>=10.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vision-forge=vision_forge.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
