"""Tests for dashboard orchestration using synthetic screenplay text only."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from services.demo_data import synthetic_screenplays
from src.evaluation.llm_client import LLMConfig
from services.pipeline import (
    RunWorkspace,
    analytics_summary,
    load_historical_results,
    normalize_stages,
    outputs_zip,
    provider_usage_summary,
    records_csv,
    run_live_evaluation,
    run_metrics,
    run_pipeline,
    validate_uploads,
)


class DashboardServiceTests(unittest.TestCase):
    def test_stage_selection_adds_prerequisites_in_pipeline_order(self) -> None:
        self.assertEqual(
            normalize_stages(["Ranking"]),
            [
                "Parsing",
                "Candidate generation",
                "Promo proposal generation",
                "Evaluation",
                "Ranking",
            ],
        )

    def test_upload_validation_supports_fifty_demo_files_and_rejects_other_content(self) -> None:
        samples = synthetic_screenplays(50)
        accepted, errors, _ = validate_uploads(samples, demo_mode=True)
        self.assertEqual(len(accepted), 50)
        self.assertEqual(errors, [])

        rejected, errors, _ = validate_uploads(
            [("episode-001-synthetic-demo.txt", b"confidential or unknown content")],
            demo_mode=True,
        )
        self.assertEqual(rejected, [])
        self.assertIn("only unchanged files from the synthetic sample pack", errors[0])

    def test_upload_validation_reports_empty_wrong_type_duplicate_and_count(self) -> None:
        duplicate_files = [
            ("same.txt", b"not empty"),
            ("same.txt", b"not empty"),
            ("empty.txt", b""),
            ("notes.md", b"not a screenplay"),
        ]
        _, errors, warnings = validate_uploads(duplicate_files, demo_mode=False)
        self.assertTrue(any("Duplicate filenames" in error for error in errors))
        self.assertTrue(any("file is empty" in error for error in errors))
        self.assertTrue(any("must be .txt" in error for error in errors))
        self.assertTrue(warnings)

        too_many = [(f"episode-{index}.txt", b"synthetic") for index in range(51)]
        accepted, errors, _ = validate_uploads(too_many, demo_mode=False)
        self.assertEqual(accepted, [])
        self.assertTrue(any("at most 50" in error for error in errors))

    def test_demo_cannot_read_historical_results(self) -> None:
        self.assertIsNone(load_historical_results(demo_mode=True, root=Path("missing-root")))

    def test_usage_summary_only_reports_provider_returned_tokens(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        workspace = RunWorkspace(temporary, Path(temporary.name))
        try:
            raw = (
                workspace.root
                / "outputs"
                / "proposal_evaluations"
                / "raw_responses"
            )
            raw.mkdir(parents=True)
            (raw / "response.json").write_text(
                json.dumps(
                    {
                        "provider_metadata": {
                            "usage": {"input_tokens": 12, "output_tokens": 8}
                        }
                    }
                ),
                encoding="utf-8",
            )
            (raw / "no-usage.json").write_text(
                json.dumps({"provider_metadata": {}}),
                encoding="utf-8",
            )

            usage = provider_usage_summary(workspace)

            self.assertEqual(usage["requests_with_usage"], 1)
            self.assertEqual(
                usage["reported_token_counts"],
                {"input_tokens": 12, "output_tokens": 8},
            )
            self.assertNotIn("estimated_cost", usage)
            self.assertEqual(
                provider_usage_summary(workspace, provider="claude"),
                {"requests_with_usage": 0, "reported_token_counts": {}},
            )
        finally:
            workspace.cleanup()

    def test_local_historical_results_are_read_only_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data" / "output"
            for relative, value in (
                ("promo_selection/ranked_189.json", {"record_count": 1, "ranked_proposals": []}),
                ("candidates/summary.json", {"counts": {"parsed_episode_files": 1}}),
                ("promo_proposals/summary.json", {"summary": {"proposal_count_before_validation": 1}}),
                (
                    "proposal_evaluations/consolidated-189/manifest.json",
                    {"validated_record_count": 1, "unvalidated_count": 0},
                ),
                (
                    "parsed/episode-001.json",
                    {"scenes": [{"events": [{"start": {"valid": True}, "end": {"valid": True}}]}]},
                ),
            ):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            (root / "README.md").write_text(
                "\n".join(
                    [
                        "## 10. Human review and final five recommendations",
                        "| Post | Episode / proposal ID | Source range(s) | Duration | Hook | Status |",
                        "|---|---|---|---|---|---|",
                        "| 1 | synthetic-demo | 00:01.00-00:21.00 | 20s | Synthetic text | Review |",
                        "## 11. AI use disclosure",
                    ]
                ),
                encoding="utf-8",
            )

            history = load_historical_results(demo_mode=False, root=root)
            self.assertIsNotNone(history)
            self.assertEqual(history["scene_count"], 1)
            self.assertEqual(history["event_count"], 1)
            self.assertEqual(history["ranking"]["record_count"], 1)
            self.assertEqual(history["final_recommendations"][0]["Source range(s)"], "00:01.00-00:21.00")

    def test_synthetic_pipeline_metrics_exports_and_cleanup(self) -> None:
        files = synthetic_screenplays(3)
        validated, errors, _ = validate_uploads(files, demo_mode=True)
        self.assertEqual(errors, [])
        workspace = run_pipeline(
            validated,
            selected_stages=["LLM input preparation", "Evaluation", "Ranking"],
            run_gemini=False,
        )
        workspace_path = workspace.root
        try:
            metrics = run_metrics(workspace)
            self.assertEqual(metrics["uploaded_episodes"], 3)
            self.assertEqual(metrics["parsed_episodes"], 3)
            self.assertEqual(metrics["scenes"], 6)
            self.assertEqual(metrics["events"], 36)
            self.assertGreater(metrics["candidates"], 0)
            self.assertGreater(metrics["proposals"], 0)
            self.assertEqual(metrics["evaluations"], 0)
            self.assertIn("LLM input preparation", workspace.stage_seconds)
            self.assertEqual(metrics["evaluation_summary"]["status"], "dry_run")

            analytics = analytics_summary(metrics)
            self.assertEqual(analytics["source"], "current_execution")
            self.assertEqual(analytics["evaluation_count"], 0)
            self.assertIn("promo_id", metrics["shortlist"][0])
            self.assertIn(b"promo_id", records_csv(metrics["results"]))

            archive = zipfile.ZipFile(BytesIO(outputs_zip(workspace)))
            names = archive.namelist()
            self.assertTrue(any(name.endswith("shortlist.json") for name in names))
            self.assertFalse(any(name.startswith("inputs/") for name in names))
            summary_path = next(name for name in names if name.endswith("proposal_dry_run_summary.json"))
            summary = json.loads(archive.read(summary_path))
            self.assertEqual(summary["api_calls"], 0)
        finally:
            workspace.cleanup()
        self.assertFalse(workspace_path.exists())

    def test_explicit_live_evaluation_is_limited_and_reports_progress(self) -> None:
        files = synthetic_screenplays(1)
        validated, errors, _ = validate_uploads(files, demo_mode=True)
        self.assertEqual(errors, [])
        workspace = run_pipeline(
            validated,
            selected_stages=["Promo proposal generation"],
            run_gemini=False,
        )
        progress = []

        class InvalidMockProvider:
            def evaluate_proposal(self, proposal):
                return "not-json", {
                    "provider": "openai",
                    "requested_model": "mock-model",
                    "returned_model": "mock-model",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }

        try:
            with patch(
                "src.evaluation.promo_evaluator.create_provider",
                return_value=InvalidMockProvider(),
            ):
                result = run_live_evaluation(
                    workspace,
                    config=LLMConfig(
                        "openai", "mock-model", "never-written-key", 1, 0, 1
                    ),
                    limit=1,
                    progress=lambda done, total, promo: progress.append(
                        (done, total, promo)
                    ),
                )
            self.assertEqual(result["expected"], 1)
            self.assertEqual(result["validated"], 0)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(len(progress), 2)
            self.assertEqual(progress[0][:2], (0, 1))
            self.assertEqual(progress[1][:2], (1, 1))
            failures_path = (
                workspace.root
                / "outputs"
                / "proposal_evaluations"
                / "failed_proposals.json"
            )
            failures = json.loads(failures_path.read_text(encoding="utf-8"))["failures"]
            self.assertEqual(failures[0]["error_type"], "EvaluationValidationError")
            usage = provider_usage_summary(workspace)
            self.assertEqual(
                usage["reported_token_counts"],
                {"prompt_tokens": 10, "completion_tokens": 2},
            )
            raw = (
                workspace.root
                / "outputs"
                / "proposal_evaluations"
                / "raw_responses"
            )
            raw_content = "\n".join(
                path.read_text(encoding="utf-8") for path in raw.glob("*.json")
            )
            self.assertNotIn("never-written-key", raw_content)
        finally:
            workspace.cleanup()


if __name__ == "__main__":
    unittest.main()
