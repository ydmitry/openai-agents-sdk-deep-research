"""
Step 4 of an AutoGPT‑style deep‑research pipeline
=================================================
Verification Layer (Critic Agent)
--------------------------------
Takes the final report produced by *Step 3* and the source documents from *Step 2*,
then performs an automated peer-review to verify factual accuracy and consistency.

Highlights
~~~~~~~~~~
* **Fact-checking** – identifies claims in the report that lack proper source support
* **Conflict detection** – flags any contradictory claims across different sources
* **Accuracy assessment** – provides an overall evaluation of the report's reliability
* **Structured feedback** – delivers actionable insights for report improvement
"""
from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from agents import Agent, Runner, RunConfig
from deep_research.step2 import Document
from deep_research.step3 import Report
from deep_research.utils import load_dotenv_files, get_model_settings

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class VerificationIssue:
    """Represents a specific issue found during verification."""
    issue_type: str  # "unsupported_claim" or "conflicting_claim"
    description: str
    location: str  # Reference to where in the report the issue appears
    severity: str  # "high", "medium", "low"
    suggested_fix: Optional[str] = None

    def to_json(self) -> str:
        """Serialize issue to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

@dataclass
class VerificationReport:
    """Contains the results of the verification process."""
    issues: List[VerificationIssue]
    accuracy_score: float  # 0.0 to 1.0
    overall_assessment: str
    
    def to_json(self) -> str:
        """Serialize verification report to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

# -----------------------------
# Critic Agent implementation
# -----------------------------

class CriticAgent:
    """
    A verification agent that reviews research reports against source documents
    to identify factual inaccuracies and contradictions.
    """

    def __init__(self, model: str = "gpt-4.1"):
        """
        Initialize the critic agent.

        Args:
            model: OpenAI model to use.
        """
        self.model = model
        self.agent = Agent(
            name="CriticAgent",
            instructions="""
            You are *CriticAgent*, tasked specifically with fact-checking research reports against source documents.
            
            Your primary responsibility is to **identify any claims in the report that lack proper source support** and **flag any conflicting claims across different sources**. This kind of agent essentially performs an automated peer-review. It won't be perfect (since the LLM might miss subtle errors), but it often catches blatant hallucinations or mistakes. For example, if the report stated a number or quote that doesn't appear in any source, the critic can flag that as "unsupported." Similarly, if two sources reported different values for something but the summary only presented one without noting the discrepancy, the critic might flag a "conflict." Having an agent explicitly look for these problems greatly increases the chance of noticing an error before it reaches the end user.
            
            For each issue you find, provide the specific claim or statement in question, where it appears in the report, why it's problematic (unsupported or contradictory), severity level (high/medium/low), and a suggested correction when possible.
            
            After analyzing all content, provide an accuracy score from 0.0 to 1.0 and an overall assessment summarizing the report's reliability.
            
            Respond with structured JSON containing all issues found and your assessment.
            """,
            tools=[]  # No external tools needed, just document analysis
        )

    async def verify_report(
        self,
        report: Report,
        source_documents: List[Document]
    ) -> VerificationReport:
        """
        Verify a research report against source documents.

        Args:
            report: The research report to verify.
            source_documents: The source documents used to create the report.

        Returns:
            VerificationReport with identified issues and assessment.
        """
        # Prepare the verification task description
        verification_task = self._prepare_verification_task(report, source_documents)
        
        # Create run configuration
        run_config = RunConfig(
            model=self.model,
            model_settings=get_model_settings(
                model_name=self.model,
                temperature=0.0  # Zero temperature for consistent verification
            ),
            tracing_disabled=False,  # Enable tracing for verification
            workflow_name="Report Verification"
        )

        print(f"Verifying report: {report.title}")
        
        # Run the verification
        result = await Runner.run(
            self.agent,
            verification_task,
            run_config=run_config,
            max_turns=1,  # Single turn for verification
        )
        
        print(f"Verification complete")
        
        # Process the verification results
        return self._process_verification_results(result.final_output)
    
    def _prepare_verification_task(
        self,
        report: Report,
        source_documents: List[Document]
    ) -> str:
        """
        Prepare the verification task description from the report and source documents.
        
        Args:
            report: The research report to verify.
            source_documents: The source documents used to create the report.
            
        Returns:
            A formatted verification task string.
        """
        # Format report content
        report_content = (
            f"# {report.title}\n\n"
            f"## Report Content\n{report.body_md}\n\n"
        )
        
        # Add references if available
        if hasattr(report, 'references') and report.references:
            report_content += "## References\n\n"
            for ref in report.references:
                report_content += f"{ref}\n"
        
        # Format source documents
        sources_content = "\n\n".join([
            f"# Source {i+1}: {doc.title}\nURL: {doc.url}\n\n{doc.text}"
            for i, doc in enumerate(source_documents)
        ])
        
        # Combine into verification task
        task = (
            "## VERIFICATION TASK\n\n"
            "You are tasked with verifying the accuracy of the following research report "
            "against the provided source documents. Identify any unsupported claims or "
            "contradictions between sources, and provide an overall assessment.\n\n"
            "## RESEARCH REPORT TO VERIFY\n\n"
            f"{report_content}\n\n"
            "## SOURCE DOCUMENTS\n\n"
            f"{sources_content}\n\n"
            "## VERIFICATION INSTRUCTIONS\n\n"
            "Please analyze the report against the source documents and provide a structured "
            "JSON response containing:\n"
            "1. A list of issues (unsupported_claim or conflicting_claim)\n"
            "2. An accuracy score from 0.0 to 1.0\n"
            "3. An overall assessment of the report's reliability\n\n"
            "Format your response as a valid JSON object."
        )
        
        return task
    
    def _process_verification_results(self, verification_output: str) -> VerificationReport:
        """
        Process the verification results from the agent's output.
        
        Args:
            verification_output: Raw output from the verification agent.
            
        Returns:
            VerificationReport object.
        """
        try:
            # Try to extract JSON from the output (handle potential text before/after JSON)
            start_idx = verification_output.find('{')
            end_idx = verification_output.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = verification_output[start_idx:end_idx]
                verification_data = json.loads(json_str)
            else:
                # Fallback if no JSON found
                raise ValueError("No JSON found in verification output")
            
            # Extract issues
            issues = []
            for issue_data in verification_data.get("issues", []):
                issues.append(VerificationIssue(
                    issue_type=issue_data.get("issue_type", "unknown"),
                    description=issue_data.get("description", ""),
                    location=issue_data.get("location", ""),
                    severity=issue_data.get("severity", "medium"),
                    suggested_fix=issue_data.get("suggested_fix")
                ))
            
            # Create verification report
            return VerificationReport(
                issues=issues,
                accuracy_score=float(verification_data.get("accuracy_score", 0.5)),
                overall_assessment=verification_data.get("overall_assessment", "")
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Handle parsing errors by creating a minimal report
            return VerificationReport(
                issues=[VerificationIssue(
                    issue_type="system_error",
                    description=f"Failed to parse verification results: {str(e)}",
                    location="N/A",
                    severity="high"
                )],
                accuracy_score=0.0,
                overall_assessment="Verification failed due to processing error"
            )

# -----------------------------
# Public API
# -----------------------------

async def async_verify_report(
    report: Report,
    source_documents: List[Document],
    model: str = "gpt-4.1"
) -> VerificationReport:
    """
    Async function to verify a research report against source documents.

    Args:
        report: Research report to verify.
        source_documents: Source documents used to create the report.
        model: OpenAI model to use.

    Returns:
        VerificationReport with issues and assessment.
    """
    critic = CriticAgent(model=model)
    return await critic.verify_report(report, source_documents)

def verify_report(
    report: Report,
    source_documents: List[Document],
    model: str = "gpt-4.1"
) -> VerificationReport:
    """
    Blocking function to verify a research report against source documents.

    Args:
        report: Research report to verify.
        source_documents: Source documents used to create the report.
        model: OpenAI model to use.

    Returns:
        VerificationReport with issues and assessment.
    """
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We are already in an event loop, so just create and return the coroutine
            return async_verify_report(
                report=report,
                source_documents=source_documents,
                model=model
            )
    except RuntimeError:
        # No event loop running, so use asyncio.run to create one
        return asyncio.run(async_verify_report(
            report=report,
            source_documents=source_documents,
            model=model
        ))

# -----------------------------
# CLI entry point
# -----------------------------

if __name__ == "__main__":
    import argparse
    import pathlib
    import sys

    # Load environment variables from .env files
    loaded_files = load_dotenv_files()

    parser = argparse.ArgumentParser(description="Step 4 – verify research report against source documents")
    parser.add_argument("report_json", help="Path to report JSON produced by Step 3")
    parser.add_argument("corpus_jsonl", help="Path to corpus JSONL produced by Step 2")
    parser.add_argument("--out", default="verification.json", help="Output JSON file (default: verification.json)")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name")
    args = parser.parse_args()

    # Load report
    report_path = pathlib.Path(args.report_json)
    if not report_path.exists():
        sys.exit(f"Report file not found: {report_path}")

    with report_path.open() as f:
        report_dict = json.load(f)

    from deep_research.step3 import Report
    report = Report(
        title=report_dict["title"],
        research_question=report_dict["research_question"],
        key_findings=report_dict["key_findings"],
        detailed_analysis=report_dict["detailed_analysis"],
        conclusions=report_dict["conclusions"]
    )

    # Load corpus
    corpus_path = pathlib.Path(args.corpus_jsonl)
    if not corpus_path.exists():
        sys.exit(f"Corpus file not found: {corpus_path}")

    documents = []
    with corpus_path.open() as f:
        for line in f:
            if line.strip():
                doc_dict = json.loads(line)
                documents.append(Document(
                    source_task_id=doc_dict["source_task_id"],
                    url=doc_dict["url"],
                    title=doc_dict["title"],
                    text=doc_dict["text"]
                ))

    # Run verification
    verification = verify_report(
        report,
        documents,
        model=args.model
    )

    # Write output
    with open(args.out, "w", encoding="utf-8") as out_f:
        out_f.write(verification.to_json())

    print(f"Wrote verification report → {args.out}")
    print(f"Found {len(verification.issues)} issues. Overall accuracy score: {verification.accuracy_score:.2f}") 