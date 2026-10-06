"""Unit tests for LLMReportExtractor."""

import json
import pandas as pd
import pytest

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.nlp.llm_extractor import LLMReportExtractor


@pytest.fixture
def extractor():
    return LLMReportExtractor()


def test_build_extraction_prompt(extractor):
    """Verify prompt formatting contains instructions and target schema."""
    text = "Complete ACL tear with intact meniscus."
    prompt = extractor.build_extraction_prompt(text)

    assert "Expected JSON schema:" in prompt
    assert text in prompt
    for target in DEFAULT_TARGET_COLS:
        assert target in prompt


def test_parse_llm_response_valid_json(extractor):
    """Verify parsing clean JSON response maps to all 12 classes bounded in [0.0, 1.0]."""
    mock_payload = {target: 0.85 if "acl" in target else 0.05 for target in DEFAULT_TARGET_COLS}
    json_str = json.dumps(mock_payload)

    parsed = extractor.parse_llm_response(json_str)
    assert len(parsed) == len(DEFAULT_TARGET_COLS)
    assert parsed["acl"] == pytest.approx(0.85)
    assert parsed["pcl"] == pytest.approx(0.05)


def test_parse_llm_response_markdown_fences(extractor):
    """Verify parsing JSON wrapped in markdown code fences and conversational text."""
    mock_payload = {target: 0.10 for target in DEFAULT_TARGET_COLS}
    mock_payload["acl"] = 0.95
    fenced_str = (
        "Here is the parsed evaluation according to the criteria:\n\n"
        "```json\n"
        f"{json.dumps(mock_payload, indent=2)}\n"
        "```\n\n"
        "Hope this helps!"
    )

    parsed = extractor.parse_llm_response(fenced_str)
    assert parsed["acl"] == pytest.approx(0.95)
    assert parsed["joint_effusion"] == pytest.approx(0.10)


def test_parse_llm_response_clamping_and_missing(extractor):
    """Verify out-of-range probabilities are clamped to [0.0, 1.0] and missing keys filled."""
    incomplete_payload = {
        "acl": 1.5,  # Needs clamping to 1.0
        "pcl": -0.2,  # Needs clamping to 0.0
    }
    parsed = extractor.parse_llm_response(json.dumps(incomplete_payload))

    assert parsed["acl"] == 1.0
    assert parsed["pcl"] == 0.0
    # Missing target should get default prior
    assert parsed["mcl"] == pytest.approx(extractor.default_prior)


def test_extract_with_fallback_success(extractor):
    """Verify successful LLM function execution."""
    def mock_llm(prompt: str) -> str:
        data = {c: 0.1 for c in DEFAULT_TARGET_COLS}
        data["acl"] = 0.92
        return json.dumps(data)

    res = extractor.extract_with_fallback("ACL tear", llm_fn=mock_llm)
    assert res["acl"] == pytest.approx(0.92)


def test_extract_with_fallback_error(extractor):
    """Verify graceful fallback to ClinicalReportParser when LLM call fails."""
    def broken_llm(prompt: str) -> str:
        raise RuntimeError("API Rate Limit Exceeded")

    report = "Complete full-thickness tear of the anterior cruciate ligament (ACL)."
    res = extractor.extract_with_fallback(report, llm_fn=broken_llm)

    # Should fall back to rule-based parser and detect ACL tear
    assert res["acl"] >= 0.90


def test_extract_batch(extractor):
    """Verify batch extraction over a DataFrame."""
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1"],
            "Report": ["Intact ACL and PCL. Moderate joint effusion."],
        }
    )
    res_df = extractor.extract_batch(df, report_col="Report")

    assert len(res_df) == 1
    assert res_df.loc[0, "acl"] <= 0.05
    assert res_df.loc[0, "joint_effusion"] >= 0.70
