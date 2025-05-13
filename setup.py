from setuptools import setup, find_packages

setup(
    name="deep-research",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.12.0",
        "openai-agents>=0.0.14",
        "python-dotenv>=1.0.0",
        "beautifulsoup4>=4.12.0",
        "aiohttp>=3.8.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.*",
            "pytest-mock==3.11.*",
            "pytest-asyncio>=0.21.0",
        ],
    },
    python_requires=">=3.8",
    description="A multi-step deep-research pipeline using the OpenAI Agents SDK",
    author="Your Name",
    author_email="your.email@example.com",
) 