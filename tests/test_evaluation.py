"""Tests for the evaluation module."""

import json

from auto_research.evaluation import EvaluationRecord, EvaluationStore


class TestEvaluationRecord:
    def test_creation_with_defaults(self):
        r = EvaluationRecord(
            eval_id="test-001",
            eval_type="evo_validate",
            baseline_score=0.5,
            candidate_score=0.8,
            passed=True,
        )
        assert r.eval_id == "test-001"
        assert r.timestamp  # auto-filled
        assert r.passed is True

    def test_auto_id_generation(self):
        r = EvaluationRecord(
            eval_id="",
            eval_type="quality_gate",
            baseline_score=0.3,
            candidate_score=0.6,
            passed=True,
        )
        assert r.eval_id.startswith("eval-")

    def test_to_dict_roundtrip(self):
        r = EvaluationRecord(
            eval_id="rt-1",
            eval_type="evo_validate",
            baseline_score=0.4,
            candidate_score=0.9,
            passed=True,
            artifact_paths=["/tmp/val.json"],
            metadata={"candidate": "my-skill"},
        )
        d = r.to_dict()
        r2 = EvaluationRecord.from_dict(d)
        assert r2.eval_id == "rt-1"
        assert r2.candidate_score == 0.9
        assert r2.artifact_paths == ["/tmp/val.json"]
        assert r2.metadata["candidate"] == "my-skill"

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "eval_id": "x",
            "eval_type": "evo_validate",
            "baseline_score": 0.1,
            "candidate_score": 0.2,
            "passed": False,
            "unknown_field": "ignored",
        }
        r = EvaluationRecord.from_dict(d)
        assert r.eval_id == "x"
        assert not hasattr(r, "unknown_field") or "unknown_field" not in r.to_dict()


class TestEvaluationStore:
    def test_save_and_load(self, tmp_path):
        store = EvaluationStore(tmp_path / "evals")
        r = EvaluationRecord(
            eval_id="s1",
            eval_type="evo_validate",
            baseline_score=0.5,
            candidate_score=0.8,
            passed=True,
        )
        path = store.save(r)
        assert path.exists()
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].eval_id == "s1"

    def test_load_filters_by_type(self, tmp_path):
        store = EvaluationStore(tmp_path / "evals")
        store.save(EvaluationRecord("a", "evo_validate", 0.5, 0.8, True))
        store.save(EvaluationRecord("b", "quality_gate", 0.3, 0.6, True))
        evo = store.load_all(eval_type="evo_validate")
        assert len(evo) == 1
        assert evo[0].eval_id == "a"

    def test_load_empty_dir(self, tmp_path):
        store = EvaluationStore(tmp_path / "nonexistent")
        assert store.load_all() == []

    def test_check_regression_no_history(self, tmp_path):
        store = EvaluationStore(tmp_path / "evals")
        result = store.check_regression("evo_validate", 0.7)
        assert result["has_baseline"] is False
        assert result["regression"] is False

    def test_check_regression_detects_drop(self, tmp_path):
        store = EvaluationStore(tmp_path / "evals")
        store.save(EvaluationRecord("h1", "evo_validate", 0.5, 0.9, True))
        result = store.check_regression("evo_validate", 0.7)
        assert result["has_baseline"] is True
        assert result["regression"] is True
        assert result["best_historical"] == 0.9
        assert result["delta"] < 0

    def test_check_regression_no_drop(self, tmp_path):
        store = EvaluationStore(tmp_path / "evals")
        store.save(EvaluationRecord("h1", "evo_validate", 0.5, 0.7, True))
        result = store.check_regression("evo_validate", 0.8)
        assert result["regression"] is False
        assert result["delta"] > 0
