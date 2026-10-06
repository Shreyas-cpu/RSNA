"""Clinical Radiology Report Parser using NegEx and Lexical Scoping.

Extracts calibrated pseudo-probabilities in [0.0, 1.0] across all 12 target
knee abnormalities from unstructured radiology narratives.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.evaluation.split import DEFAULT_TARGET_COLS

# Default probability calibration constants
P_DEFINITIVE_POSITIVE = 0.95
P_MODERATE_POSITIVE = 0.75
P_EQUIVOCAL = 0.50
P_TRACE = 0.20
P_DEFAULT_PRIOR = 0.10
P_DEFINITIVE_NEGATIVE = 0.05


class ClinicalReportParser:
    """Deterministic NegEx rule-based parser for knee MRI radiology impressions."""

    # Target class definitions with regex patterns
    TARGET_PATTERNS = {
        "acl": {
            "anatomy": re.compile(
                r"\b(acl|anterior\s+cruciate\s+ligament|(?<!posterior\s)cruciate\s+ligaments?)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(tear|rupture|disruption|attenuat(ed|ion)|sprain|avulsion|torn)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|continuous|preserved|no\s+tear)\b",
                re.IGNORECASE,
            ),
        },
        "pcl": {
            "anatomy": re.compile(
                r"\b(pcl|posterior\s+cruciate\s+ligament|(?<!anterior\s)cruciate\s+ligaments?)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(tear|rupture|disruption|attenuat(ed|ion)|sprain|torn)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|continuous|preserved|no\s+tear)\b",
                re.IGNORECASE,
            ),
        },
        "mcl": {
            "anatomy": re.compile(
                r"\b(mcl|medial\s+collateral\s+ligament|(?<!lateral\s)collateral\s+ligaments?)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(sprain|tear|rupture|disruption|edema|thickening|torn)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|continuous|preserved|no\s+sprain|no\s+tear)\b",
                re.IGNORECASE,
            ),
        },
        "lcl": {
            "anatomy": re.compile(
                r"\b(lcl|lateral\s+collateral\s+ligament|fibular\s+collateral\s+ligament|(?<!medial\s)collateral\s+ligaments?)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(sprain|tear|rupture|disruption|edema|thickening|torn)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|continuous|preserved|no\s+sprain|no\s+tear)\b",
                re.IGNORECASE,
            ),
        },
        "medial_meniscus": {
            "anatomy": re.compile(
                r"\b(medial\s+menisc(us|al)|posterior\s+horn\s+of\s+(the\s+)?medial\s+meniscus|anterior\s+horn\s+of\s+(the\s+)?medial\s+meniscus|body\s+of\s+(the\s+)?medial\s+meniscus|(?<!lateral\s)menisci)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(tear|macerat(ed|ion)|truncat(ed|ion)|fraying|complex\s+tear|horizontal\s+cleavage|radial\s+tear|bucket[-\s]handle|torn)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|no\s+tear|preserved)\b",
                re.IGNORECASE,
            ),
        },
        "lateral_meniscus": {
            "anatomy": re.compile(
                r"\b(lateral\s+menisc(us|al)|posterior\s+horn\s+of\s+(the\s+)?lateral\s+meniscus|anterior\s+horn\s+of\s+(the\s+)?lateral\s+meniscus|body\s+of\s+(the\s+)?lateral\s+meniscus|(?<!medial\s)menisci)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(tear|macerat(ed|ion)|truncat(ed|ion)|fraying|complex\s+tear|horizontal\s+cleavage|radial\s+tear|bucket[-\s]handle|torn)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|no\s+tear|preserved)\b",
                re.IGNORECASE,
            ),
        },
        "patellar_tendon": {
            "anatomy": re.compile(
                r"\b(patellar\s+tendon|patellar\s+ligament)\b", re.IGNORECASE
            ),
            "abnormal": re.compile(
                r"\b(tendinosis|tendinopathy|tear|thickening|edema|disruption|partial\s+tear)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|continuous|preserved|no\s+tear)\b",
                re.IGNORECASE,
            ),
        },
        "quadriceps_tendon": {
            "anatomy": re.compile(
                r"\b(quadriceps\s+tendon|quad\s+tendon)\b", re.IGNORECASE
            ),
            "abnormal": re.compile(
                r"\b(tendinosis|tendinopathy|tear|thickening|edema|disruption|partial\s+tear|rupture)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(intact|normal|unremarkable|continuous|preserved|no\s+tear)\b",
                re.IGNORECASE,
            ),
        },
        "cartilage_femur": {
            "anatomy": re.compile(
                r"\b(femoral\s+(condyle|trochlea)|cartilage\s+of\s+the\s+femur|femoral\s+cartilage|medial\s+femoral|lateral\s+femoral|femoral\s+joint)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(chondromalacia|cartilage\s+(loss|thinning|defect|wear|fissuring)|osteoarthritis|osteophyte(s)?|sclerosis|defect)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(normal\s+cartilage|preserved|intact|unremarkable|normal)\b",
                re.IGNORECASE,
            ),
        },
        "cartilage_tibia": {
            "anatomy": re.compile(
                r"\b(tibial\s+plateau|cartilage\s+of\s+the\s+tibia|tibial\s+cartilage|medial\s+tibial|lateral\s+tibial|tibial\s+joint)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(chondromalacia|cartilage\s+(loss|thinning|defect|wear|fissuring)|osteoarthritis|osteophyte(s)?|sclerosis|defect)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(normal\s+cartilage|preserved|intact|unremarkable|normal)\b",
                re.IGNORECASE,
            ),
        },
        "cartilage_patella": {
            "anatomy": re.compile(
                r"\b(patellar\s+cartilage|patella(r)?\s+facet|retropatellar|patellofemoral|trochlear\s+groove)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(chondromalacia|cartilage\s+(loss|thinning|defect|wear|fissuring)|osteoarthritis|fissuring|defect)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(normal\s+cartilage|preserved|intact|unremarkable|normal)\b",
                re.IGNORECASE,
            ),
        },
        "joint_effusion": {
            "anatomy": re.compile(
                r"\b(joint\s+effusion|effusion|suprapatellar\s+(effusion|fluid)|fluid\s+collection|fluid\s+distension)\b",
                re.IGNORECASE,
            ),
            "abnormal": re.compile(
                r"\b(effusion|fluid\s+collection|distension|accumulation|large|moderate|small|present|increased)\b",
                re.IGNORECASE,
            ),
            "normal": re.compile(
                r"\b(no\s+effusion|without\s+effusion|physiological\s+fluid|trace\s+fluid|no\s+significant\s+fluid|normal)\b",
                re.IGNORECASE,
            ),
        },
    }

    # Map external or alternative labels to canonical patterns
    ALIAS_MAP = {
        "acl": "acl",
        "ACL": "acl",
        "pcl": "pcl",
        "PCL": "pcl",
        "mcl": "mcl",
        "MCL": "mcl",
        "lcl": "lcl",
        "LCL": "lcl",
        "medial_meniscus": "medial_meniscus",
        "Medial Meniscus": "medial_meniscus",
        "lateral_meniscus": "lateral_meniscus",
        "Lateral Meniscus": "lateral_meniscus",
        "patellar_tendon": "patellar_tendon",
        "Patellar Tendon": "patellar_tendon",
        "quadriceps_tendon": "quadriceps_tendon",
        "Quadriceps Tendon": "quadriceps_tendon",
        "cartilage_femur": "cartilage_femur",
        "Cartilage Femur": "cartilage_femur",
        "Medial OA": "cartilage_femur",
        "cartilage_tibia": "cartilage_tibia",
        "Cartilage Tibia": "cartilage_tibia",
        "Lateral OA": "cartilage_tibia",
        "cartilage_patella": "cartilage_patella",
        "Cartilage Patella": "cartilage_patella",
        "PF OA": "cartilage_patella",
        "joint_effusion": "joint_effusion",
        "Joint Effusion": "joint_effusion",
        "Effusion": "joint_effusion",
        "Synovitis": "joint_effusion",
        "Baker's": "joint_effusion",
        "Contusion": "cartilage_tibia",
        "Fracture": "cartilage_femur",
    }

    # Negation trigger patterns
    PRE_NEGATION_PATTERN = re.compile(
        r"\b(no\s+(evidence|sign|discrete|focal|acute|definite)?\s*(of)?|"
        r"negative\s+for|without(\s+evidence\s+of)?|free\s+of|rules?\s+out|"
        r"not\s+(seen|identified|observed|detected)|absence\s+of)\b",
        re.IGNORECASE,
    )

    POST_NEGATION_PATTERN = re.compile(
        r"\b(is\s+(intact|normal|unremarkable|preserved)|are\s+(intact|normal|unremarkable|preserved)|"
        r"intact|unremarkable|normal|preserved|continuous|"
        r"shows?\s+no\s+(tear|fracture|abnormality|sprain)|"
        r"within\s+normal\s+limits|wnl)\b",
        re.IGNORECASE,
    )

    # Certainty modifiers
    HIGH_CERTAINTY_PATTERN = re.compile(
        r"\b(complete|full[-\s]thickness|high[-\s]grade|gross|severe|marked|macerat(ed|ion)|torn|disrupt(ed|ion))\b",
        re.IGNORECASE,
    )

    EQUIVOCAL_PATTERN = re.compile(
        r"\b(cannot\s+be\s+(ruled\s+out|excluded)|possible|suspected|questionable|equivocal|indeterminate|suggestive\s+of)\b",
        re.IGNORECASE,
    )

    TRACE_PATTERN = re.compile(
        r"\b(trace|physiological|minimal|subtle)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        target_cols: Optional[List[str]] = None,
        default_prior: float = P_DEFAULT_PRIOR,
    ):
        """Initialize the ClinicalReportParser.

        Args:
            target_cols: List of target pathology names to extract.
            default_prior: Default probability when pathology is unmentioned.
        """
        self.target_cols = target_cols or DEFAULT_TARGET_COLS
        self.default_prior = default_prior

    def _split_clauses(self, text: str) -> List[str]:
        """Split text into sentences and clinical clauses."""
        if not text or not isinstance(text, str):
            return []
        # Split on sentence terminals, semicolons, numbered bullets, or newlines
        raw_parts = re.split(r"[.;:\n\r]+|\d+\.\s+", text)
        return [p.strip() for p in raw_parts if p.strip()]

    def parse_clause(self, clause: str, target: str) -> Optional[Tuple[str, float]]:
        """Parse a single clause for a specific target abnormality.

        Returns:
            Tuple of (status, probability) or None if target is not mentioned.
            status: 'negative', 'positive', or 'equivocal'
        """
        canonical_target = self.ALIAS_MAP.get(target, target.lower())
        patterns = self.TARGET_PATTERNS.get(canonical_target) or self.TARGET_PATTERNS.get(target)
        if not patterns:
            return None

        # Check if target anatomy/entity is referenced in this clause
        anatomy_match = patterns["anatomy"].search(clause)
        if not anatomy_match:
            return None

        # Check for explicit normal/intact phrase
        normal_match = patterns["normal"].search(clause)
        post_neg = self.POST_NEGATION_PATTERN.search(clause)
        pre_neg = self.PRE_NEGATION_PATTERN.search(clause)

        # Check for abnormality mention
        abnormal_match = patterns["abnormal"].search(clause)

        # 1. Clear negation or normal finding
        if pre_neg or post_neg or normal_match:
            # If pre-negation is present before abnormality or target
            if pre_neg:
                # E.g., "no evidence of ACL tear" -> negative
                return ("negative", P_DEFINITIVE_NEGATIVE)
            if post_neg or normal_match:
                # E.g., "ACL is intact", "ACL is unremarkable"
                return ("negative", P_DEFINITIVE_NEGATIVE)

        # 2. Positive finding described
        if abnormal_match:
            # Check certainty modifiers
            if self.EQUIVOCAL_PATTERN.search(clause):
                return ("equivocal", P_EQUIVOCAL)
            elif self.TRACE_PATTERN.search(clause):
                return ("trace", P_TRACE)
            elif self.HIGH_CERTAINTY_PATTERN.search(clause):
                return ("positive", P_DEFINITIVE_POSITIVE)
            else:
                return ("positive", P_MODERATE_POSITIVE)

        # Target mentioned without explicit positive or negative (neutral mention)
        return ("neutral", self.default_prior)

    def parse_report(self, text: str) -> Dict[str, float]:
        """Parse full radiology report into calibrated probabilities for all targets.

        Args:
            text: Unstructured radiology impression string.

        Returns:
            Dict mapping each target name to a continuous float in [0.0, 1.0].
        """
        if not text or not isinstance(text, str):
            return {target: self.default_prior for target in self.target_cols}

        clauses = self._split_clauses(text)
        results: Dict[str, float] = {}

        for target in self.target_cols:
            target_findings: List[Tuple[str, float]] = []

            for clause in clauses:
                finding = self.parse_clause(clause, target)
                if finding is not None:
                    target_findings.append(finding)

            if not target_findings:
                # Not mentioned anywhere in the report
                results[target] = self.default_prior
            else:
                # Determine overall probability for this target
                # Priority: If any clause describes a definitive positive tear/fracture,
                # adopt highest positive probability unless explicitly refuted.
                statuses = [f[0] for f in target_findings]
                probs = [f[1] for f in target_findings]

                if "positive" in statuses or "equivocal" in statuses:
                    # Choose max probability among positive/equivocal findings
                    results[target] = max(probs)
                elif "negative" in statuses:
                    # Explicitly cleared as intact/normal
                    results[target] = P_DEFINITIVE_NEGATIVE
                else:
                    results[target] = probs[0]

        return results

    def parse_dataframe(
        self,
        df: pd.DataFrame,
        report_col: str = "Report",
        output_prefix: str = "",
    ) -> pd.DataFrame:
        """Parse an entire dataframe of reports into 12 pseudo-label columns.

        Args:
            df: Input dataframe containing report narratives.
            report_col: Name of the column with text impressions.
            output_prefix: Optional prefix for target column names.

        Returns:
            New DataFrame with study IDs and continuous pseudo-probability columns.
        """
        parsed_records = []
        for text in df[report_col].fillna(""):
            probs = self.parse_report(str(text))
            if output_prefix:
                probs = {f"{output_prefix}{k}": v for k, v in probs.items()}
            parsed_records.append(probs)

        parsed_df = pd.DataFrame(parsed_records, index=df.index)
        return parsed_df
