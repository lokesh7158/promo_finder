from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.promo_construction.promo_proposal_generator import generate_proposals


def _timestamp(seconds: int) -> str:
    return f"00:{seconds:02d}.00"


def _episode_documents(episode_number: int) -> tuple[dict, dict]:
    episode_id = f"episode-{episode_number:03d}"
    scene_id = f"{episode_id}:scene-001"
    events = []
    for index, (start, end, event_type) in enumerate(
        [
            (0, 5, "CHARACTER_DIALOGUE"),
            (5, 10, "ACTION"),
            (10, 15, "CHARACTER_DIALOGUE"),
            (15, 20, "ACTION"),
            (20, 25, "CHARACTER_DIALOGUE"),
            (25, 30, "CHARACTER_DIALOGUE"),
        ],
        start=1,
    ):
        speaker = "MIRA" if event_type == "CHARACTER_DIALOGUE" else None
        event = {
            "id": f"{episode_id}:scene-001:event-{index:03d}",
            "type": event_type,
            "speaker_label_raw": speaker,
            "annotations_raw": [],
            "start": {"raw": _timestamp(start)},
            "end": {"raw": _timestamp(end)},
            "content": {"kind": "dialogue" if speaker else "description", "raw": f"Beat {index}."},
            "validation_flags": [],
        }
        events.append({"scene_id": scene_id, "event": event})

    scene = {
        "id": scene_id,
        "number": 1,
        "heading": {"raw": "INT. ROOM - DAY"},
        "events": [wrapped["event"] for wrapped in events],
    }
    parsed = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "metadata": {
            "episode_number": {"value": f"Episode {episode_number}"},
            "episode_logline": {"value": ""},
            "continuity_notes": {"value": ""},
        },
        "scenes": [scene],
    }
    candidate = {
        "candidate_id": f"{episode_id}:clip-0001",
        "episode": {"episode_id": episode_id, "episode_number": episode_number},
        "time_range": {
            "start": {"raw": "00:00.00"},
            "end": {"raw": "00:30.00"},
            "duration_seconds": 30.0,
        },
        "events": events,
    }
    candidate_document = {"schema_version": "1.0", "candidates": [candidate]}
    return parsed, candidate_document


class PromoProposalGeneratorTests(unittest.TestCase):
    def _write_fixture(self, root: Path, episode_numbers: tuple[int, ...]) -> tuple[Path, Path]:
        parsed_dir = root / "parsed"
        candidates_dir = root / "candidates"
        parsed_dir.mkdir()
        candidates_dir.mkdir()
        for episode_number in episode_numbers:
            parsed, candidate_doc = _episode_documents(episode_number)
            (parsed_dir / f"{parsed['episode_id']}.json").write_text(
                json.dumps(parsed), encoding="utf-8"
            )
            (candidates_dir / f"{parsed['episode_id']}.json").write_text(
                json.dumps(candidate_doc), encoding="utf-8"
            )
        return parsed_dir, candidates_dir

    def test_builds_source_traceable_single_and_multi_episode_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parsed_dir, candidates_dir = self._write_fixture(
                Path(temporary_directory), (1, 2)
            )
            result = generate_proposals(parsed_dir, candidates_dir)

        categories = {item["promo_category"] for item in result["all_proposals"]}
        self.assertIn("SINGLE_SEGMENT", categories)
        self.assertIn("SAME_EPISODE_MULTI_SEGMENT", categories)
        self.assertIn("CROSS_EPISODE_MULTI_SEGMENT", categories)
        self.assertGreater(result["summary"]["valid_proposal_count"], 0)
        for proposal in result["all_proposals"]:
            self.assertTrue(proposal["source_event_ids"])
            self.assertTrue(proposal["segments"])
            for segment in proposal["segments"]:
                self.assertEqual(
                    len(segment["source_event_ids"]),
                    len(set(segment["source_event_ids"])),
                )
                self.assertTrue(segment["events"])
                self.assertEqual(
                    set(segment["source_event_ids"]),
                    {wrapped["source_event_ref"] for wrapped in segment["events"]},
                )
        for proposal in result["shortlist"]:
            self.assertEqual(proposal["validation_status"], "VALID")
            self.assertGreaterEqual(proposal["total_duration_seconds"], 20)
            self.assertLessEqual(proposal["total_duration_seconds"], 60)

    def test_output_is_deterministic_and_preserves_episode_scoped_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parsed_dir, candidates_dir = self._write_fixture(
                Path(temporary_directory), (1, 2)
            )
            first = generate_proposals(parsed_dir, candidates_dir)
            second = generate_proposals(parsed_dir, candidates_dir)

        first_ids = [item["promo_id"] for item in first["all_proposals"]]
        second_ids = [item["promo_id"] for item in second["all_proposals"]]
        self.assertEqual(first_ids, second_ids)
        for proposal in first["all_proposals"]:
            for segment in proposal["segments"]:
                self.assertTrue(
                    all(
                        event_ref.startswith(f"{segment['episode_id']}:")
                        for event_ref in segment["source_event_ids"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
