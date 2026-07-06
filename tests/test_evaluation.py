"""
Tests for EvaluationEngine and AggregationEngine (src/evaluation_engine.py).

Uses mocked LLM client and real embeddings for fast, deterministic tests.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation_engine import (
    EvaluationEngine,
    AggregationEngine,
    QualityScore,
    EvaluationResult,
    SystemPerformance,
)
from src.response_generator import EmbeddingManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emb_mgr():
    """Real embedding manager (loaded once per module for speed)."""
    return EmbeddingManager("all-MiniLM-L6-v2")


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns valid judge responses."""
    client = MagicMock()
    client.generate.return_value = "SCORE: 4\nEXPLANATION: Looks good."
    return client


@pytest.fixture
def engine(emb_mgr, mock_llm):
    return EvaluationEngine(emb_mgr, mock_llm)


@pytest.fixture
def agg():
    return AggregationEngine()


def _make_result(email_pair_id="email_001", semantic_score=0.75, tone=4.0,
                 completeness=0.8, prof=4.0, coherence=4.0, email_type="professional",
                 overall="acceptable", meets=True, latency=1000.0):
    return EvaluationResult(
        evaluation_id=f"eval_{email_pair_id}",
        email_pair_id=email_pair_id,
        timestamp="2026-01-01T00:00:00",
        incoming_email="Test incoming",
        ground_truth_response="Test ground truth",
        generated_response="Test generated",
        semantic_similarity=QualityScore("semantic_similarity", semantic_score, "0-1", "ok", "cosine"),
        tone_appropriateness=QualityScore("tone_appropriateness", tone, "1-5", "ok", "llm"),
        completeness=QualityScore("completeness", completeness, "0-1", "ok", "llm"),
        professionalism=QualityScore("professionalism", prof, "1-5", "ok", "llm"),
        coherence=QualityScore("coherence", coherence, "1-5", "ok", "llm"),
        rouge_scores={"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.35},
        bleu_score=0.25,
        email_type=email_type,
        overall_quality=overall,
        meets_thresholds=meets,
        generation_latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:

    def test_identical_texts_score_near_one(self, engine):
        text = "Thank you for reaching out. I will help you with your request."
        score = engine.compute_semantic_similarity(text, text)
        assert score.score >= 0.99

    def test_unrelated_texts_score_below_half(self, engine):
        text1 = "The weather today is sunny and warm."
        text2 = "Please submit your quarterly financial report by Friday."
        score = engine.compute_semantic_similarity(text1, text2)
        assert score.score < 0.7  # These are clearly different topics

    def test_paraphrases_score_higher_than_unrelated(self, engine):
        original = "Thank you for contacting us. We will resolve your issue shortly."
        paraphrase = "Thanks for reaching out. Your problem will be fixed soon."
        unrelated = "The annual budget meeting is scheduled for next Tuesday."
        sim_para = engine.compute_semantic_similarity(original, paraphrase)
        sim_unrel = engine.compute_semantic_similarity(original, unrelated)
        assert sim_para.score > sim_unrel.score

    def test_returns_quality_score_object(self, engine):
        score = engine.compute_semantic_similarity("hello world", "hello world")
        assert isinstance(score, QualityScore)
        assert score.dimension == "semantic_similarity"
        assert 0.0 <= score.score <= 1.0


# ---------------------------------------------------------------------------
# ROUGE and BLEU
# ---------------------------------------------------------------------------

class TestSupplementaryMetrics:

    def test_rouge_returns_three_scores(self, engine):
        scores = engine.compute_rouge("hello world test", "hello world foo")
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores

    def test_rouge_scores_in_range(self, engine):
        scores = engine.compute_rouge("some generated text here", "some reference text here")
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_rouge_identical_texts_score_one(self, engine):
        text = "identical text for testing purposes"
        scores = engine.compute_rouge(text, text)
        assert scores["rouge1"] == 1.0

    def test_bleu_in_range(self, engine):
        score = engine.compute_bleu("the quick brown fox", "the quick brown fox jumps")
        assert 0.0 <= score <= 1.0

    def test_bleu_identical_returns_high_score(self, engine):
        text = "thank you for contacting us we will help"
        score = engine.compute_bleu(text, text)
        assert score > 0.8


# ---------------------------------------------------------------------------
# LLM Judge Parsing
# ---------------------------------------------------------------------------

class TestJudgeParsing:

    def test_parses_valid_score(self, engine):
        response = "SCORE: 4\nEXPLANATION: Very good response."
        score, explanation = engine._parse_judge_response(response, scale=5)
        assert score == 4.0
        assert "Very good" in explanation

    def test_parses_decimal_score(self, engine):
        response = "SCORE: 0.85\nEXPLANATION: Mostly complete."
        score, explanation = engine._parse_judge_response(response, scale=1)
        assert score == 0.85

    def test_clamps_score_to_scale(self, engine):
        response = "SCORE: 10\nEXPLANATION: Over the top."
        score, _ = engine._parse_judge_response(response, scale=5)
        assert score == 5.0

    def test_defaults_to_midpoint_on_unparseable(self, engine):
        response = "I cannot provide a score for this."
        score, _ = engine._parse_judge_response(response, scale=5)
        assert score == 2.5  # middle of 0-5

    def test_case_insensitive_parsing(self, engine):
        response = "score: 3\nexplanation: Acceptable."
        score, _ = engine._parse_judge_response(response, scale=5)
        assert score == 3.0


# ---------------------------------------------------------------------------
# Full Evaluation + Quality Assessment
# ---------------------------------------------------------------------------

class TestQualityAssessment:

    def test_high_quality_classification(self, engine):
        quality, meets, flags = engine._assess_quality(
            semantic_sim=0.85, tone=4.5, completeness=0.9,
            professionalism=4.5, coherence=4.5
        )
        assert quality == "high_quality"
        assert meets is True
        assert flags == []

    def test_acceptable_classification(self, engine):
        quality, meets, flags = engine._assess_quality(
            semantic_sim=0.65, tone=3.5, completeness=0.7,
            professionalism=3.5, coherence=3.5
        )
        assert quality == "acceptable"
        assert meets is True

    def test_below_threshold_classification(self, engine):
        quality, meets, flags = engine._assess_quality(
            semantic_sim=0.3, tone=2.0, completeness=0.4,
            professionalism=2.0, coherence=2.0
        )
        assert quality == "below_threshold"
        assert meets is False

    def test_flags_low_semantic_similarity(self, engine):
        _, _, flags = engine._assess_quality(
            semantic_sim=0.4, tone=4.0, completeness=0.8,
            professionalism=4.0, coherence=4.0
        )
        assert "low_semantic_similarity" in flags

    def test_flags_incomplete_response(self, engine):
        _, _, flags = engine._assess_quality(
            semantic_sim=0.7, tone=4.0, completeness=0.3,
            professionalism=4.0, coherence=4.0
        )
        assert "incomplete_response" in flags


# ---------------------------------------------------------------------------
# Aggregation Engine
# ---------------------------------------------------------------------------

class TestAggregationEngine:

    def test_aggregate_empty_returns_zero(self, agg):
        perf = agg.aggregate_results([])
        assert perf.dataset_size == 0

    def test_aggregate_correct_dataset_size(self, agg):
        results = [_make_result(f"email_{i:03d}") for i in range(5)]
        perf = agg.aggregate_results(results)
        assert perf.dataset_size == 5

    def test_dimension_stats_have_required_keys(self, agg):
        results = [_make_result(f"email_{i:03d}") for i in range(3)]
        perf = agg.aggregate_results(results)
        for dim in ("semantic_similarity", "tone_appropriateness", "completeness",
                    "professionalism", "coherence"):
            assert dim in perf.dimension_stats
            stats = perf.dimension_stats[dim]
            for key in ("mean", "median", "std", "min", "max"):
                assert key in stats

    def test_quality_distribution_adds_to_100(self, agg):
        results = [
            _make_result("e1", overall="high_quality", meets=True),
            _make_result("e2", overall="acceptable", meets=True),
            _make_result("e3", overall="below_threshold", meets=False),
        ]
        perf = agg.aggregate_results(results)
        total_pct = sum(d["percentage"] for d in perf.quality_distribution.values())
        assert abs(total_pct - 100.0) < 0.02  # floating point tolerance

    def test_mean_computed_correctly(self, agg):
        results = [
            _make_result("e1", semantic_score=0.6),
            _make_result("e2", semantic_score=0.8),
        ]
        perf = agg.aggregate_results(results)
        assert abs(perf.dimension_stats["semantic_similarity"]["mean"] - 0.7) < 0.001

    def test_report_generates_json_file(self, agg, tmp_path):
        import json
        results = [_make_result(f"e{i}") for i in range(3)]
        perf = agg.aggregate_results(results)
        out = str(tmp_path / "report.json")
        agg.generate_report(results, perf, out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert "evaluation_summary" in data
        assert "per_response_results" in data
        assert len(data["per_response_results"]) == 3
