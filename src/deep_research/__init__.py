"""
Deep Research package - multi-stage research pipeline with OpenAI Agents.
"""

from deep_research.step1 import generate_research_plan, ResearchPlan, SubTask
from deep_research.step2 import build_corpus, Document
from deep_research.utils import load_dotenv_files

__version__ = "0.1.0" 