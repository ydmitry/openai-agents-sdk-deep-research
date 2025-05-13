from setuptools import setup, find_packages

setup(
    name="deep-research",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.12.0",
        "openai-agents>=0.0.14",
    ],
    python_requires=">=3.8",
    description="A multi-step deep-research pipeline using the OpenAI Agents SDK",
    author="Your Name",
    author_email="your.email@example.com",
) 