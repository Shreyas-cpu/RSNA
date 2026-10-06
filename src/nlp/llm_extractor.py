"""Structured LLM Clinical Report Extractor.

Prompts an LLM to extract continuous, calibrated pseudo-probabilities across all 12
knee abnormalities from unstructured radiology narratives with deterministic fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.nlp.report_parser import ClinicalReportParser, P_DEFAULT_PRIOR


class LLMReportExtractor:
    """Structured extractor prompting LLMs for calibrated 12-class pseudo-labels."""

    SYSTEM_INSTRUCTIONS = (
        "You are an expert musculoskeletal radiologist specializing in knee MRI evaluation.\n"
        "Your task is to analyze the provided knee MRI radiology impression and determine\n"
        "the diagnostic probability (confidence between 0.00 and 1.00) for each of the 12\n"
        "target conditions.\n\n"
        "Calibration Guidelines:\n"
        "- 0.90 - 1.00: Definitive abnormality (e.g., complete/full-thickness tear, marked osteoarthritis, large effusion)\n"
        "- 0.70 - 0.89: Probable / moderate abnormality (e.g., partial tear, moderate thinning, sprain)\n"
        "- 0.40 - 0.60: Equivocal / possible / suspected / indeterminate abnormality\n"
        "- 0.15 - 0.30: Trace / subtle finding or mild irregularity without definite tear\n"
        "- 0.00 - 0.05: Definitive normal / intact structure or explicit negative finding\n"
        "- 0.10: Structure not mentioned (background prior probability)\n\n"
        "Respond ONLY with a valid JSON object mapping each class name to its float probability."
    )

    def __init__(
        self,
        target_cols: Optional[List[str]] = None,
        default_prior: float = P_DEFAULT_PRIOR,
    ):
        """Initialize the LLMReportExtractor.

        Args:
            target_cols: List of 12 target column names.
            default_prior: Fallback prior when keys are missing.
        """
        self.target_cols = target_cols or DEFAULT_TARGET_COLS
        self.default_prior = default_prior
        self.parser = ClinicalReportParser(
            target_cols=self.target_cols, default_prior=default_prior
        )

    def build_extraction_prompt(self, report_text: str) -> str:
        """Format the full extraction prompt for an LLM."""
        targets_formatted = ",\n  ".join([f'"{c}": float' for c in self.target_cols])
        prompt = (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"Expected JSON schema:\n"
            f"{{\n  {targets_formatted}\n}}\n\n"
            f"Radiology Impression:\n"
            f"\"\"\"\n"
            f"{report_text.strip() if report_text else ''}\n"
            f"\"\"\"\n\n"
            f"JSON Output:"
        )
        return prompt

    def parse_llm_response(self, response_text: str) -> Dict[str, float]:
        """Clean markdown fences, parse JSON response, and validate probabilities.

        Args:
            response_text: Raw string response from an LLM.

        Returns:
            Dict mapping all target column names to floats in [0.0, 1.0].
        """
        if not response_text or not isinstance(response_text, str):
            raise ValueError("Empty or invalid LLM response text")

        # Strip markdown json code fences if present
        clean_text = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text)

        # Extract the outermost JSON object if surrounded by preamble/postscript
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        data = json.loads(clean_text)
        if not isinstance(data, dict):
            raise ValueError(f"Parsed JSON is not a dictionary: {type(data)}")

        results: Dict[str, float] = {}
        # Support case-insensitive key lookups
        lower_data = {str(k).lower().replace(" ", "_"): float(v) for k, v in data.items() if self._is_number(v)}

        for target in self.target_cols:
            norm_target = target.lower().replace(" ", "_")
            if target in data and self._is_number(data[target]):
                val = float(data[target])
            elif norm_target in lower_data:
                val = lower_data[norm_target]
            else:
                # Missing key -> fall back to default prior
                val = self.default_prior

            # Clip strictly to [0.0, 1.0]
            results[target] = float(np.clip(val, 0.0, 1.0))

        return results

    @staticmethod
    def _is_number(val: Any) -> bool:
        """Check if a value can be converted to float."""
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def extract_with_fallback(
        self,
        report_text: str,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, float]:
        """Extract probabilities using LLM with automatic fallback to NegEx parser.

        Args:
            report_text: Radiology report text.
            llm_fn: Optional callable taking a prompt string and returning LLM response string.

        Returns:
            Dict mapping each target class to float probability in [0.0, 1.0].
        """
        if llm_fn is not None and callable(llm_fn):
            try:
                prompt = self.build_extraction_prompt(report_text)
                response = llm_fn(prompt)
                return self.parse_llm_response(response)
            except Exception:
                # Graceful fallback to deterministic rule parser
                pass

        # Fallback to NegEx rule parser
        return self.parser.parse_report(report_text)

    def extract_batch(
        self,
        df: pd.DataFrame,
        report_col: str = "Report",
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> pd.DataFrame:
        """Extract pseudo-labels for a DataFrame using LLM with deterministic fallback."""
        records = []
        for text in df[report_col].fillna(""):
            res = self.extract_with_fallback(str(text), llm_fn=llm_fn)
            records.append(res)
        return pd.DataFrame(records, index=df.index)
