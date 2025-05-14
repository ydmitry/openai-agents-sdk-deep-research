"""
AutoGPT-style deep-research pipeline
===================================
An end-to-end pipeline for deep research on any topic.
"""

from deep_research.step1 import generate_research_plan
from deep_research.step2 import build_corpus
from deep_research.step3 import generate_report
from deep_research.step4 import verify_report
from deep_research.utils import load_dotenv_files

__version__ = "0.1.0" 