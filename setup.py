from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="company-clasterization",
    version="0.1.0",
    author="Mikhail M.",
    author_email="@datanalist",
    description="Проект для кластеризации компаний",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/company-clasterization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires="==3.11.*",
    install_requires=[
        "fastapi>=0.115.12",
        "uvicorn>=0.34.0",
        "jose>=1.0.0",
        "bcrypt>=4.3.0",
        "pydantic>=2.11.2",
        "pandas>=2.2.3",
        "python-dotenv>=1.1.0",
        "scikit-learn>=1.6.1",
        "dagster>=1.10.11",
        "dagster-webserver>=1.10.11",
        "dagster-duckdb>=0.26.11",
        "sentence-transformers>=4.1.0",
        "plotly>=6.0.1",
        "bertopic>=0.17.0",
        "hdbscan>=0.8.40",
        "nltk>=3.9.1",
        "keybert>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pre-commit>=4.2.0",
            "pyupgrade>=3.19.1",
            "check-jsonschema>=0.32.1",
            "detect-secrets>=1.5.0",
            "ipykernel>=6.29.5",
            "pytest>=8.3.5",
        ],
    },
)
