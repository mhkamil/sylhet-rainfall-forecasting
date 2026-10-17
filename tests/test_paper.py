"""Tests for the Stage-9 manuscript checker (src/paper.py). Synthetic tables in tmp_path; only the last tests read the
real draft and bibliography. Run: python -m pytest -q tests/test_paper.py"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paper as P  # noqa: E402


@pytest.fixture
def tables(tmp_path):
    pd.DataFrame({"model": ["a", "b"], "BSS": [0.37555, -0.2632], "flag": [True, False]}).to_csv(tmp_path / "t.csv",
                                                                                                 index=False)
    pd.DataFrame({"quantity": ["share"], "value": [0.5566]}).to_csv(tmp_path / "d.csv", index=False)
    return tmp_path


def check(md, tables):
    return P.check_sources(md, tables).iloc[0]


def test_correctly_rounded_number_in_text_passes(tables):
    r = check("BSS was 0.376. <!-- source: t.csv | model=a | BSS=0.376 -->", tables)
    assert r.value_ok and r.in_text


def test_wrong_number_and_missing_text_are_flagged(tables):
    assert not check("BSS was 0.381. <!-- source: t.csv | model=a | BSS=0.381 -->", tables).value_ok
    r = check("BSS was high. <!-- source: t.csv | model=a | BSS=0.376 -->", tables)
    assert r.value_ok and not r.in_text


def test_minus_sign_abs_percent_bool_and_long_tables(tables):
    assert check("ΔMAE −0.263 <!-- source: t.csv | model=b | BSS=-0.263 -->", tables).value_ok
    assert check("worse by 0.263 <!-- source: t.csv | model=b | BSS=abs(0.263) -->", tables).value_ok
    assert check("55.7 % wet <!-- source: d.csv | * | share=55.7% -->", tables).value_ok
    assert check("not significant <!-- source: t.csv | model=b | flag=False -->", tables).value_ok


def test_untraced_numbers_and_unhedged_significance():
    md = "# 4 Results\n\nSkill was 0.42 and 0.30. <!-- source: t.csv | model=a | BSS=0.42 -->\n"
    assert P.untraced_numbers(md).number.tolist() == ["0.30"]
    sig = P.significance_claims("# 5 Discussion\n\nModel X was significantly better.\n\nY was not significant.\n")
    assert len(sig) == 1 and not sig.ok.iloc[0]


def test_abbreviation_must_be_defined_at_first_use():
    md = "We use the BSS here. Later the Brier skill score (BSS) and the critical success index (CSI) and CSI."
    assert P.abbreviations(md).abbreviation.tolist() == ["BSS"]


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_latex_keeps_source_comments(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("Text 0.5 [@duan1983]. <!-- source: t.csv | model=a | BSS=0.5 -->\n\n<!-- BEGIN TABLE 1 -->\n")
    tex = P.to_latex(md).read_text()
    assert "% source: t.csv | model=a | BSS=0.5" in tex and "\\citep{duan1983}" in tex and "BEGIN TABLE" not in tex


def test_draft_cites_only_the_verified_bibliography():
    keys = set(P.bib_keys())
    assert len(keys) == 17                                         # 16 verified references; DM and HLN separately
    assert set(P.citation_keys(P.DRAFT.read_text())) == keys
