"""Unit tests for ClinicalReportParser."""

import pandas as pd
import pytest

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.nlp.report_parser import (
    ClinicalReportParser,
    P_DEFINITIVE_NEGATIVE,
    P_DEFINITIVE_POSITIVE,
    P_EQUIVOCAL,
)


@pytest.fixture
def parser():
    return ClinicalReportParser()


def test_parser_all_12_classes_returned(parser):
    """Verify parser returns all 12 target classes bounded strictly in [0.0, 1.0]."""
    text = "Knee MRI shows normal appearance of all structures."
    res = parser.parse_report(text)

    assert isinstance(res, dict)
    assert len(res) == len(DEFAULT_TARGET_COLS)
    for col in DEFAULT_TARGET_COLS:
        assert col in res
        assert 0.0 <= res[col] <= 1.0


def test_parser_definitive_positive_and_negative(parser):
    """Verify positive tear detection vs intact/normal assertions."""
    text = (
        "1. Complete full-thickness tear of the anterior cruciate ligament (ACL) with retraction.\n"
        "2. The posterior cruciate ligament and MCL are intact.\n"
        "3. Patellar tendon and quadriceps tendon are normal."
    )
    res = parser.parse_report(text)

    # acl should be high certainty positive
    assert res["acl"] >= P_DEFINITIVE_POSITIVE
    # pcl and mcl should be definitive negative
    assert res["pcl"] <= P_DEFINITIVE_NEGATIVE
    assert res["mcl"] <= P_DEFINITIVE_NEGATIVE
    assert res["patellar_tendon"] <= P_DEFINITIVE_NEGATIVE


def test_parser_negex_scoping(parser):
    """Verify pre-negation scopes across meniscal and ligament structures."""
    text = (
        "IMPRESSION:\n"
        "No evidence of medial meniscus tear or lateral meniscus tear.\n"
        "No joint effusion or femoral cartilage defect.\n"
        "Tibial plateau cartilage is preserved."
    )
    res = parser.parse_report(text)

    assert res["medial_meniscus"] <= P_DEFINITIVE_NEGATIVE
    assert res["lateral_meniscus"] <= P_DEFINITIVE_NEGATIVE
    assert res["joint_effusion"] <= P_DEFINITIVE_NEGATIVE
    assert res["cartilage_femur"] <= P_DEFINITIVE_NEGATIVE
    assert res["cartilage_tibia"] <= P_DEFINITIVE_NEGATIVE


def test_parser_equivocal_hedging(parser):
    """Verify uncertainty markers map to intermediate probability."""
    text = (
        "Medial collateral ligament (MCL) shows periligamentous signal, "
        "and a possible low-grade sprain cannot be excluded."
    )
    res = parser.parse_report(text)

    assert res["mcl"] == pytest.approx(P_EQUIVOCAL, abs=0.05)


def test_parser_cartilage_and_effusion(parser):
    """Verify cartilage thinning and joint effusion extraction."""
    text = (
        "Marked femoral condyle cartilage wear and thinning. "
        "Patellar cartilage shows mild chondromalacia. "
        "Large joint effusion present."
    )
    res = parser.parse_report(text)

    assert res["cartilage_femur"] >= 0.75
    assert res["cartilage_patella"] >= 0.70
    assert res["joint_effusion"] >= 0.75


def test_parser_custom_target_aliases():
    """Verify parser operates seamlessly with custom or title-case competition target names."""
    custom_targets = ["ACL", "MCL", "Medial Meniscus", "Effusion", "PF OA"]
    custom_parser = ClinicalReportParser(target_cols=custom_targets)
    text = "Complete tear of ACL. Intact MCL. No effusion. Mild patellofemoral chondromalacia."

    res = custom_parser.parse_report(text)
    assert res["ACL"] >= 0.90
    assert res["MCL"] <= 0.05
    assert res["Effusion"] <= 0.05
    assert res["PF OA"] >= 0.70


def test_parser_empty_or_none_text(parser):
    """Verify handling of empty or None reports returns default prior."""
    res_none = parser.parse_report(None)
    res_empty = parser.parse_report("")

    for col in DEFAULT_TARGET_COLS:
        assert res_none[col] == parser.default_prior
        assert res_empty[col] == parser.default_prior


def test_parser_dataframe(parser):
    """Verify batch parsing over pandas DataFrame."""
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s2"],
            "Report": [
                "Complete ACL tear. Intact menisci.",
                "Normal study with intact cruciate ligaments and no effusion.",
            ],
        }
    )
    res_df = parser.parse_dataframe(df, report_col="Report")

    assert len(res_df) == 2
    assert list(res_df.columns) == DEFAULT_TARGET_COLS
    assert res_df.loc[0, "acl"] >= 0.90
    assert res_df.loc[1, "acl"] <= 0.05
