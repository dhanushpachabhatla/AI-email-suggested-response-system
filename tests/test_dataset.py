"""
Tests for DatasetManager (src/dataset_manager.py).

Covers: synthetic generation, validation, loading, statistics.
"""

import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_manager import DatasetManager, EmailPair, EmailMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    return DatasetManager()


def _make_valid_pair(pair_id="email_001", incoming_len=80, response_len=80):
    return {
        "id": pair_id,
        "incoming_email": "I" * incoming_len,
        "response": "R" * response_len,
        "metadata": {
            "subject": "Test Subject",
            "formality_level": "formal",
            "email_type": "professional",
            "subject_category": "inquiry",
            "sender_role": "customer",
        },
    }


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

class TestSyntheticDataGeneration:

    def test_generates_correct_size(self, manager):
        pairs = manager.generate_synthetic_dataset(size=20)
        assert len(pairs) == 20

    def test_generates_100_pairs(self, manager):
        pairs = manager.generate_synthetic_dataset(size=100)
        assert len(pairs) == 100

    def test_distribution_customer_support(self, manager):
        pairs = manager.generate_synthetic_dataset(size=100)
        cs = [p for p in pairs if p.metadata.email_type == "customer_support"]
        assert 25 <= len(cs) <= 35  # ~30%

    def test_distribution_professional(self, manager):
        pairs = manager.generate_synthetic_dataset(size=100)
        prof = [p for p in pairs if p.metadata.email_type == "professional"]
        assert 35 <= len(prof) <= 45  # ~40%

    def test_distribution_technical(self, manager):
        pairs = manager.generate_synthetic_dataset(size=100)
        tech = [p for p in pairs if p.metadata.email_type == "technical"]
        assert 25 <= len(tech) <= 35  # ~30%

    def test_all_pairs_have_required_fields(self, manager):
        pairs = manager.generate_synthetic_dataset(size=10)
        for p in pairs:
            assert p.id
            assert p.incoming_email
            assert p.response
            assert p.metadata.formality_level in ("formal", "semi-formal", "casual")
            assert p.metadata.email_type in ("customer_support", "professional", "technical")

    def test_ids_are_unique(self, manager):
        pairs = manager.generate_synthetic_dataset(size=50)
        ids = [p.id for p in pairs]
        assert len(ids) == len(set(ids))

    def test_saves_to_file(self, manager, tmp_path):
        path = str(tmp_path / "dataset.json")
        manager.generate_synthetic_dataset(size=10, filepath=path)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["total_pairs"] == 10
        assert len(data["email_pairs"]) == 10


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_pair_passes(self, manager):
        is_valid, msg = manager.validate_email_pair(_make_valid_pair())
        assert is_valid is True
        assert msg == ""

    def test_missing_id_fails(self, manager):
        pair = _make_valid_pair()
        del pair["id"]
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False
        assert "id" in msg.lower()

    def test_incoming_too_short_fails(self, manager):
        pair = _make_valid_pair(incoming_len=10)
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False
        assert "short" in msg.lower()

    def test_incoming_too_long_fails(self, manager):
        pair = _make_valid_pair(incoming_len=2001)
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False
        assert "long" in msg.lower()

    def test_response_too_short_fails(self, manager):
        pair = _make_valid_pair(response_len=10)
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False

    def test_response_too_long_fails(self, manager):
        pair = _make_valid_pair(response_len=1501)
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False

    def test_invalid_formality_level_fails(self, manager):
        pair = _make_valid_pair()
        pair["metadata"]["formality_level"] = "ultra-formal"
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False

    def test_invalid_email_type_fails(self, manager):
        pair = _make_valid_pair()
        pair["metadata"]["email_type"] = "spam"
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False

    def test_invalid_sender_role_fails(self, manager):
        pair = _make_valid_pair()
        pair["metadata"]["sender_role"] = "robot"
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False

    def test_empty_incoming_after_strip_fails(self, manager):
        pair = _make_valid_pair()
        pair["incoming_email"] = "   "
        is_valid, msg = manager.validate_email_pair(pair)
        assert is_valid is False


# ---------------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:

    def test_loads_valid_json(self, manager, tmp_path):
        path = str(tmp_path / "ds.json")
        manager.generate_synthetic_dataset(size=10, filepath=path)
        pairs = manager.load_dataset(path)
        assert len(pairs) == 10
        assert all(isinstance(p, EmailPair) for p in pairs)

    def test_raises_on_missing_file(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            manager.load_dataset(str(tmp_path / "missing.json"))

    def test_raises_when_no_valid_pairs(self, manager, tmp_path):
        bad_data = {"email_pairs": [{"id": "x", "incoming_email": "short", "response": "short", "metadata": {}}]}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_data))
        with pytest.raises(ValueError):
            manager.load_dataset(str(path))

    def test_skips_invalid_pairs_and_loads_valid(self, manager, tmp_path):
        valid = _make_valid_pair("email_001")
        invalid = {"id": "email_002", "incoming_email": "too short", "response": "R" * 80,
                   "metadata": {"subject": "X", "formality_level": "formal", "email_type": "professional",
                                "subject_category": "inquiry", "sender_role": "customer"}}
        data = {"email_pairs": [valid, invalid]}
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps(data))
        pairs = manager.load_dataset(str(path))
        assert len(pairs) == 1
        assert pairs[0].id == "email_001"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestDatasetStatistics:

    def test_returns_correct_total(self, manager):
        manager.generate_synthetic_dataset(size=20)
        stats = manager.get_dataset_statistics()
        assert stats["total_pairs"] == 20

    def test_formality_distribution_keys(self, manager):
        manager.generate_synthetic_dataset(size=30)
        stats = manager.get_dataset_statistics()
        dist = stats["formality_level_distribution"]["counts"]
        assert set(dist.keys()).issubset({"formal", "semi-formal", "casual"})

    def test_email_type_distribution_keys(self, manager):
        manager.generate_synthetic_dataset(size=30)
        stats = manager.get_dataset_statistics()
        dist = stats["email_type_distribution"]["counts"]
        assert set(dist.keys()).issubset({"customer_support", "professional", "technical"})

    def test_length_stats_have_required_keys(self, manager):
        manager.generate_synthetic_dataset(size=10)
        stats = manager.get_dataset_statistics()
        for key in ("min", "max", "mean", "median"):
            assert key in stats["incoming_email_length_stats"]
            assert key in stats["response_length_stats"]

    def test_empty_dataset_returns_zero(self, manager):
        stats = manager.get_dataset_statistics()
        assert stats["total_pairs"] == 0
