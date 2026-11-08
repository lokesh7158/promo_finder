from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.evaluation.evaluation_prompts import (
    PROPOSAL_SYSTEM_PROMPT,
    build_proposal_prompt,
)
from src.evaluation.evaluation_schema import (
    EvaluationValidationError,
    calculate_final_score,
    classification_for_score,
    gemini_proposal_response_schema,
    normalize_proposal,
    validate_proposal_evaluation,
)
from src.evaluation.llm_client import (
    GeminiProvider,
    LLMConfig,
    LLMQuotaExceededError,
)
from src.evaluation.promo_evaluator import (
    load_proposals,
    run_proposal_evaluations,
)


def _ts(seconds: int) -> str:
    return f"00:{seconds:02d}.00"


def _ts_hundredths(value: int) -> str:
    minutes, remainder = divmod(value, 6_000)
    seconds, hundredths = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{hundredths:02d}"


def _fixture(
    category: str,
    episode_numbers: tuple[int, ...] | None = None,
) -> tuple[dict, dict, dict]:
    if episode_numbers is None:
        episode_numbers = (1,) if category == "SINGLE_SEGMENT" else (1, 2)
    if category == "SINGLE_SEGMENT":
        segment_specs = [(episode_numbers[0], 0, 20)]
    elif category == "SAME_EPISODE_MULTI_SEGMENT":
        segment_specs = [(episode_numbers[0], 0, 10), (episode_numbers[0], 20, 30)]
    elif category == "CROSS_EPISODE_MULTI_SEGMENT":
        segment_specs = [(episode_numbers[0], 0, 10), (episode_numbers[1], 0, 10)]
    else:
        raise AssertionError(category)

    source_events: dict[tuple[str, str], dict] = {}
    source_candidates: dict[str, dict] = {}
    segments: list[dict] = []
    proposal_event_ids: list[str] = []
    proposal_candidate_ids: list[str] = []
    for order, (episode_number, start, end) in enumerate(segment_specs, start=1):
        episode_id = f"episode-{episode_number:03d}"
        scene_id = f"{episode_id}:scene-001"
        candidate_id = f"{episode_id}:clip-{order:04d}"
        events = []
        segment_event_ids = []
        for event_start in range(start, end, 5):
            event_index = event_start // 5 + 1
            event_end = min(event_start + 5, end)
            event_id = f"{episode_id}:scene-001:event-{event_index:03d}"
            text = "Mira enters" if event_index == 1 else (
                "Door closes" if event_end == end else f"Beat {event_index}"
            )
            event = {
                "id": event_id,
                "type": "ACTION",
                "speaker_label_raw": None,
                "annotations_raw": [],
                "start": {"raw": _ts(event_start)},
                "end": {"raw": _ts(event_end)},
                "content": {"kind": "description", "raw": text},
                "validation_flags": [],
            }
            wrapped = {
                "scene_id": scene_id,
                "source_event_ref": event_id,
                "source_event_id": event_id,
                "event": event,
            }
            events.append(wrapped)
            segment_event_ids.append(event_id)
            source_events[(episode_id, event_id)] = {
                "event": event,
                "episode_number": episode_number,
            }
        source_candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "episode": {
                "episode_id": episode_id,
                "episode_number": episode_number,
            },
            "events": events,
        }
        segment = {
            "segment_id": f"promo-test:segment-{order:02d}",
            "segment_order": order,
            "episode_id": episode_id,
            "episode_number": episode_number,
            "start_timestamp": _ts(start),
            "end_timestamp": _ts(end),
            "duration_seconds": end - start,
            "playback_offset_seconds": sum(
                prior_end - prior_start
                for _, prior_start, prior_end in segment_specs[: order - 1]
            ),
            "playback_end_seconds": sum(
                prior_end - prior_start
                for _, prior_start, prior_end in segment_specs[:order]
            ),
            "source_candidate_ids": [candidate_id],
            "source_event_ids": segment_event_ids,
            "events": events,
        }
        segments.append(segment)
        proposal_event_ids.extend(
            event_id for event_id in segment_event_ids if event_id not in proposal_event_ids
        )
        proposal_candidate_ids.append(candidate_id)

    proposal = {
        "promo_id": "promo-test",
        "promo_category": category,
        "segments": segments,
        "total_duration_seconds": sum(end - start for _, start, end in segment_specs),
        "source_candidate_ids": proposal_candidate_ids,
        "source_event_ids": proposal_event_ids,
        "narrative_rationale": "Fixture rationale for structural evaluation.",
        "validation_status": "VALID",
    }
    return proposal, source_events, source_candidates


def _normalized(category: str) -> dict:
    proposal, source_events, source_candidates = _fixture(category)
    return normalize_proposal(
        proposal,
        source_events=source_events,
        source_candidates=source_candidates,
    )


def _normalized_boundary_case(
    promo_id: str,
    category: str,
    *,
    segment_index: int,
    segment_start: int,
    event_start: int,
    event_end: int,
    event_id: str | None = None,
    episode_numbers: tuple[int, ...] | None = None,
) -> dict:
    proposal, source_events, source_candidates = _fixture(category, episode_numbers)
    proposal["promo_id"] = promo_id
    segment = proposal["segments"][segment_index]
    original_duration = round(segment["duration_seconds"] * 100)
    segment["segment_id"] = f"{promo_id}:segment-{segment_index + 1:02d}"
    segment["start_timestamp"] = _ts_hundredths(segment_start)
    segment["end_timestamp"] = _ts_hundredths(segment_start + original_duration)
    segment["duration_seconds"] = original_duration / 100

    wrapped = segment["events"][0]
    event = wrapped["event"]
    if event_id is not None:
        old_event_id = event["id"]
        episode_id = segment["episode_id"]
        source_record = source_events.pop((episode_id, old_event_id))
        event["id"] = event_id
        wrapped["source_event_id"] = event_id
        wrapped["source_event_ref"] = event_id
        source_events[(episode_id, event_id)] = source_record
    event["start"]["raw"] = _ts_hundredths(event_start)
    event["end"]["raw"] = _ts_hundredths(event_end)
    event["content"]["raw"] = "Boundary-crossing dramatic beat"
    segment["events"] = [wrapped]
    segment["source_event_ids"] = [event["id"]]

    playback_cursor = 0
    for current in proposal["segments"]:
        duration = round(current["duration_seconds"] * 100)
        current["playback_offset_seconds"] = playback_cursor / 100
        current["playback_end_seconds"] = (playback_cursor + duration) / 100
        playback_cursor += duration
    proposal["total_duration_seconds"] = playback_cursor / 100
    proposal["source_event_ids"] = list(
        dict.fromkeys(
            event_id
            for current in proposal["segments"]
            for event_id in current["source_event_ids"]
        )
    )
    return normalize_proposal(
        proposal,
        source_events=source_events,
        source_candidates=source_candidates,
    )


def _citation(
    proposal: dict,
    *,
    segment_index: int,
    event_index: int,
    endpoint: str,
) -> dict:
    segment = proposal["segments"][segment_index]
    event_record = segment["events"][event_index]
    source_timestamp = event_record[
        "source_start_timestamp" if endpoint == "start" else "source_end_timestamp"
    ]
    source_start = int(segment["source_start_timestamp"][3:5]) * 100
    source_value = int(source_timestamp[3:5]) * 100
    playback_value = round(segment["playback_start_offset_seconds"] * 100) + (
        source_value - source_start
    )
    playback_timestamp = _ts(playback_value // 100)
    content = event_record["event"]["content"]["raw"]
    return {
        "segment_id": segment["segment_id"],
        "playback_order": segment["playback_order"],
        "episode_id": segment["episode_id"],
        "event_id": event_record["event_id"],
        "source_timestamp": source_timestamp,
        "playback_timestamp": playback_timestamp,
        "excerpt": content,
    }


def _evaluation(proposal: dict) -> dict:
    hook_citation = _citation(proposal, segment_index=0, event_index=0, endpoint="start")
    final_segment_index = len(proposal["segments"]) - 1
    final_segment = proposal["segments"][final_segment_index]
    final_event_index = len(final_segment["events"]) - 1
    cliff_citation = _citation(
        proposal,
        segment_index=final_segment_index,
        event_index=final_event_index,
        endpoint="end",
    )
    scores = {
        "hook_strength": 4,
        "emotional_impact": 4,
        "cliffhanger_strength": 4,
        "standalone_comprehensibility": 4,
        "story_coherence": 4,
        "promotional_potential": 4,
    }
    score = calculate_final_score(scores)
    return {
        "promo_id": proposal["promo_id"],
        "promo_category": proposal["promo_category"],
        "total_duration_seconds": proposal["total_duration_seconds"],
        "scores": scores,
        "hook": {
            "description": "Mira enters",
            "within_first_3_seconds": True,
            "evidence": [hook_citation],
        },
        "emotional_impact_reasoning": "The sequence creates a clear emotional progression.",
        "cliffhanger": {
            "description": "Door closes",
            "evidence": [cliff_citation],
        },
        "standalone_comprehensibility_reasoning": "The presented events can be followed.",
        "story_coherence_reasoning": "The assembled transitions are considered.",
        "promotional_potential_reasoning": "The sequence has a clear promotional premise.",
        "strengths": ["Clear opening and ending beats."],
        "weaknesses": ["The transitions may need review."],
        "human_review_required": False,
        "confidence": "medium",
        "uncertainties": [],
        "final_score": score,
        "classification": classification_for_score(score),
    }


class SegmentAwareEvaluationTests(unittest.TestCase):
    def test_normalizes_single_same_episode_and_cross_episode_proposals(self) -> None:
        for category in (
            "SINGLE_SEGMENT",
            "SAME_EPISODE_MULTI_SEGMENT",
            "CROSS_EPISODE_MULTI_SEGMENT",
        ):
            with self.subTest(category=category):
                proposal = _normalized(category)
                self.assertEqual(proposal["promo_category"], category)
                self.assertTrue(proposal["segments"])
                self.assertEqual(
                    proposal["total_duration_seconds"],
                    proposal["segments"][-1]["playback_end_offset_seconds"],
                )

    def test_playback_offsets_and_source_to_playback_mapping(self) -> None:
        proposal = _normalized("CROSS_EPISODE_MULTI_SEGMENT")
        first, second = proposal["segments"]
        self.assertEqual(first["playback_start_offset_seconds"], 0)
        self.assertEqual(first["playback_end_offset_seconds"], 10)
        self.assertEqual(second["playback_start_offset_seconds"], 10)
        self.assertEqual(second["playback_end_offset_seconds"], 20)
        self.assertEqual(second["source_start_timestamp"], "00:00.00")
        mapped = _citation(proposal, segment_index=1, event_index=0, endpoint="start")
        self.assertEqual(mapped["source_timestamp"], "00:00.00")
        self.assertEqual(mapped["playback_timestamp"], "00:10.00")

    def test_all_four_reported_boundary_failures_have_endpoint_specific_mappings(self) -> None:
        cases = (
            (
                "promo-cross_episode_multi_segment-ac779fbe3ace",
                "CROSS_EPISODE_MULTI_SEGMENT",
                0,
                4_780,
                4_720,
                4_880,
                "episode-013:scene-002:event-004",
                (13, 14),
                None,
                "00:01.00",
            ),
            (
                "promo-cross_episode_multi_segment-6ed2078ffddd",
                "CROSS_EPISODE_MULTI_SEGMENT",
                0,
                4_780,
                4_700,
                5_000,
                "episode-013:scene-002:event-003",
                (13, 14),
                None,
                "00:02.20",
            ),
            (
                "promo-single_segment-811c14ad27bc",
                "SINGLE_SEGMENT",
                0,
                6_350,
                6_300,
                6_450,
                "episode-003:scene-002:event-036",
                (3,),
                None,
                "00:01.00",
            ),
            (
                "promo-same_episode_multi_segment-402438892301",
                "SAME_EPISODE_MULTI_SEGMENT",
                0,
                350,
                210,
                420,
                "episode-003:scene-001:event-002",
                (3, 3),
                None,
                "00:00.70",
            ),
        )
        for (
            promo_id,
            category,
            segment_index,
            segment_start,
            event_start,
            event_end,
            event_id,
            episode_numbers,
            expected_start_playback,
            expected_end_playback,
        ) in cases:
            with self.subTest(promo_id=promo_id):
                proposal = _normalized_boundary_case(
                    promo_id,
                    category,
                    segment_index=segment_index,
                    segment_start=segment_start,
                    event_start=event_start,
                    event_end=event_end,
                    event_id=event_id,
                    episode_numbers=episode_numbers,
                )
                event = proposal["segments"][segment_index]["events"][0]
                self.assertFalse(event["source_start_endpoint_eligible"])
                self.assertIsNone(event["playback_timestamp_for_source_start"])
                self.assertEqual(
                    event["playback_timestamp_for_source_start"],
                    expected_start_playback,
                )
                self.assertTrue(event["source_end_endpoint_eligible"])
                self.assertEqual(
                    event["playback_timestamp_for_source_end"],
                    expected_end_playback,
                )
                self.assertEqual(
                    event["clipped_playback_overlap_start_timestamp"],
                    proposal["segments"][segment_index]["playback_start_timestamp"],
                )

    def test_endpoint_mapping_accounts_for_nonzero_multi_segment_playback_offset(self) -> None:
        proposal = _normalized_boundary_case(
            "promo-cross-episode-second-segment-boundary",
            "CROSS_EPISODE_MULTI_SEGMENT",
            segment_index=1,
            segment_start=4_780,
            event_start=4_720,
            event_end=4_880,
            event_id="episode-002:scene-002:event-004",
        )
        event = proposal["segments"][1]["events"][0]
        self.assertFalse(event["source_start_endpoint_eligible"])
        self.assertIsNone(event["playback_timestamp_for_source_start"])
        self.assertTrue(event["source_end_endpoint_eligible"])
        self.assertEqual(event["playback_timestamp_for_source_end"], "00:11.00")

    def test_event_clipped_at_both_segment_boundaries_has_no_eligible_endpoint(self) -> None:
        proposal = _normalized_boundary_case(
            "promo-event-clipped-at-both-boundaries",
            "SINGLE_SEGMENT",
            segment_index=0,
            segment_start=350,
            event_start=210,
            event_end=2_500,
        )
        event = proposal["segments"][0]["events"][0]
        self.assertFalse(event["source_start_endpoint_eligible"])
        self.assertFalse(event["source_end_endpoint_eligible"])
        self.assertIsNone(event["playback_timestamp_for_source_start"])
        self.assertIsNone(event["playback_timestamp_for_source_end"])
        self.assertEqual(event["clipped_playback_overlap_start_timestamp"], "00:00.00")
        self.assertEqual(event["clipped_playback_overlap_end_timestamp"], "00:20.00")

    def test_valid_opening_and_ending_evidence_and_semantic_statuses(self) -> None:
        proposal = _normalized("CROSS_EPISODE_MULTI_SEGMENT")
        validated = validate_proposal_evaluation(_evaluation(proposal), proposal)
        self.assertEqual(validated["hook"]["semantic_evidence_status"], "SUPPORTED")
        self.assertEqual(validated["cliffhanger"]["semantic_evidence_status"], "SUPPORTED")
        self.assertFalse(validated["human_review_required"])

    def test_partial_semantic_support_requires_human_review(self) -> None:
        proposal = _normalized("SINGLE_SEGMENT")
        evaluation = _evaluation(proposal)
        evaluation["hook"]["description"] = "Mira enters the room"
        validated = validate_proposal_evaluation(evaluation, proposal)
        self.assertEqual(
            validated["hook"]["semantic_evidence_status"],
            "PARTIALLY_SUPPORTED",
        )
        self.assertTrue(validated["human_review_required"])

    def test_unsupported_semantic_claim_requires_human_review(self) -> None:
        proposal = _normalized("SINGLE_SEGMENT")
        evaluation = _evaluation(proposal)
        evaluation["hook"]["description"] = "A dragon appears"
        validated = validate_proposal_evaluation(evaluation, proposal)
        self.assertEqual(validated["hook"]["semantic_evidence_status"], "UNSUPPORTED")
        self.assertTrue(validated["human_review_required"])

    def test_hook_evidence_must_be_in_first_three_playback_seconds(self) -> None:
        proposal = _normalized("SINGLE_SEGMENT")
        evaluation = _evaluation(proposal)
        evaluation["hook"]["evidence"] = [
            _citation(proposal, segment_index=0, event_index=1, endpoint="end")
        ]
        with self.assertRaisesRegex(EvaluationValidationError, "hook window"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_cliffhanger_evidence_must_be_in_final_three_playback_seconds(self) -> None:
        proposal = _normalized("SAME_EPISODE_MULTI_SEGMENT")
        evaluation = _evaluation(proposal)
        evaluation["cliffhanger"]["evidence"] = [
            _citation(proposal, segment_index=0, event_index=0, endpoint="start")
        ]
        with self.assertRaisesRegex(EvaluationValidationError, "cliffhanger window"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_wrong_episode_is_rejected_even_when_source_timestamps_collide(self) -> None:
        proposal = _normalized("CROSS_EPISODE_MULTI_SEGMENT")
        evaluation = _evaluation(proposal)
        citation = evaluation["cliffhanger"]["evidence"][0]
        citation["episode_id"] = proposal["segments"][0]["episode_id"]
        with self.assertRaisesRegex(EvaluationValidationError, "segment/episode identity"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_cross_episode_timestamp_collision_does_not_match_by_time(self) -> None:
        proposal = _normalized("CROSS_EPISODE_MULTI_SEGMENT")
        evaluation = _evaluation(proposal)
        citation = evaluation["hook"]["evidence"][0]
        citation["segment_id"] = proposal["segments"][1]["segment_id"]
        citation["playback_order"] = 2
        citation["episode_id"] = proposal["segments"][1]["episode_id"]
        citation["playback_timestamp"] = "00:10.00"
        with self.assertRaisesRegex(EvaluationValidationError, "does not belong"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_wrong_segment_is_rejected(self) -> None:
        proposal = _normalized("SAME_EPISODE_MULTI_SEGMENT")
        evaluation = _evaluation(proposal)
        citation = evaluation["hook"]["evidence"][0]
        citation["segment_id"] = proposal["segments"][1]["segment_id"]
        citation["playback_order"] = 2
        citation["playback_timestamp"] = "00:10.00"
        with self.assertRaisesRegex(EvaluationValidationError, "does not belong"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_incorrect_source_timestamp_is_rejected(self) -> None:
        proposal = _normalized("SINGLE_SEGMENT")
        evaluation = _evaluation(proposal)
        evaluation["hook"]["evidence"][0]["source_timestamp"] = "00:01.00"
        with self.assertRaisesRegex(EvaluationValidationError, "not an event endpoint"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_incorrect_playback_timestamp_is_rejected(self) -> None:
        proposal = _normalized("CROSS_EPISODE_MULTI_SEGMENT")
        evaluation = _evaluation(proposal)
        evaluation["cliffhanger"]["evidence"][0]["playback_timestamp"] = "00:10.00"
        with self.assertRaisesRegex(EvaluationValidationError, "mapping is incorrect"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_invalid_event_id_and_excerpt_are_rejected(self) -> None:
        proposal = _normalized("SINGLE_SEGMENT")
        evaluation = _evaluation(proposal)
        evaluation["hook"]["evidence"][0]["event_id"] = "episode-999:event-999"
        with self.assertRaisesRegex(EvaluationValidationError, "does not belong"):
            validate_proposal_evaluation(evaluation, proposal)

        evaluation = _evaluation(proposal)
        evaluation["hook"]["evidence"][0]["excerpt"] = "not source text"
        with self.assertRaisesRegex(EvaluationValidationError, "excerpt"):
            validate_proposal_evaluation(evaluation, proposal)

    def test_prompt_explains_playback_and_source_chronology(self) -> None:
        proposal = _normalized("CROSS_EPISODE_MULTI_SEGMENT")
        prompt = json.loads(build_proposal_prompt(proposal))
        self.assertEqual(len(prompt["segments_in_playback_order"]), 2)
        self.assertEqual(
            prompt["segments_in_playback_order"][1]["playback_start_timestamp"],
            "00:10.00",
        )
        first_event = prompt["segments_in_playback_order"][0]["events_in_source_order"][0]
        self.assertIn("source_endpoint_provenance", first_event)
        self.assertIn("eligible_citation_endpoints", first_event)
        self.assertIn("clipped_playback_overlap_start_timestamp", first_event)
        self.assertIn("not original source chronology", PROPOSAL_SYSTEM_PROMPT)
        self.assertIn("narrative sense", PROPOSAL_SYSTEM_PROMPT)
        self.assertIn("clipped_playback_overlap_start_timestamp", PROPOSAL_SYSTEM_PROMPT)
        self.assertIn("NEVER valid evidence", PROPOSAL_SYSTEM_PROMPT)
        self.assertIn(
            "ONLY the separately provided eligible_citation_endpoints",
            PROPOSAL_SYSTEM_PROMPT,
        )
        self.assertIn("copied verbatim as a contiguous substring", PROPOSAL_SYSTEM_PROMPT)
        self.assertIn("Do not paraphrase, normalize punctuation", PROPOSAL_SYSTEM_PROMPT)

    def test_prompt_citation_choices_include_only_eligible_endpoints(self) -> None:
        proposal = _normalized_boundary_case(
            "promo-prompt-endpoint-eligibility",
            "SINGLE_SEGMENT",
            segment_index=0,
            segment_start=350,
            event_start=210,
            event_end=420,
        )
        prompt = json.loads(build_proposal_prompt(proposal))
        event = prompt["segments_in_playback_order"][0]["events_in_source_order"][0]

        self.assertEqual(
            event["eligible_citation_endpoints"],
            [
                {
                    "endpoint": "end",
                    "source_timestamp": "00:04.20",
                    "playback_timestamp": "00:00.70",
                }
            ],
        )
        self.assertEqual(
            event["source_endpoint_provenance"]["start"],
            {
                "source_timestamp": "00:02.10",
                "eligible": False,
                "playback_timestamp_if_eligible": None,
            },
        )
        self.assertEqual(
            event["clipped_playback_overlap_start_timestamp"],
            "00:00.00",
        )
        self.assertNotIn(
            event["clipped_playback_overlap_start_timestamp"],
            [
                endpoint["playback_timestamp"]
                for endpoint in event["eligible_citation_endpoints"]
            ],
        )

    def test_prompt_preserves_source_excerpt_characters_exactly(self) -> None:
        proposal, source_events, source_candidates = _fixture("SINGLE_SEGMENT")
        original_text = '"...still clinging to that little locket."\n'
        event = proposal["segments"][0]["events"][0]["event"]
        event["content"]["raw"] = original_text
        normalized = normalize_proposal(
            proposal,
            source_events=source_events,
            source_candidates=source_candidates,
        )
        prompt = json.loads(build_proposal_prompt(normalized))
        prompt_text = prompt["segments_in_playback_order"][0]["events_in_source_order"][0][
            "content"
        ]
        self.assertEqual(prompt_text, original_text)

    def test_dry_run_resume_and_persistence_without_provider_calls(self) -> None:
        proposal, _, source_candidates = _fixture("SINGLE_SEGMENT")
        second_proposal = deepcopy(proposal)
        second_proposal["promo_id"] = "promo-test-second"
        second_proposal["segments"][0]["segment_id"] = "promo-test-second:segment-01"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            parsed_dir = root / "parsed"
            candidates_dir = root / "candidates"
            output_dir = root / "out"
            parsed_dir.mkdir()
            candidates_dir.mkdir()
            episode_doc = {
                "episode_id": "episode-001",
                "episode_number": 1,
                "scenes": [
                    {
                        "id": "episode-001:scene-001",
                        "events": [
                            item["event"]
                            for item in source_candidates[
                                "episode-001:clip-0001"
                            ]["events"]
                        ],
                    }
                ],
            }
            (parsed_dir / "episode-001.json").write_text(
                json.dumps(episode_doc), encoding="utf-8"
            )
            candidate_doc = {
                "candidates": list(source_candidates.values()),
            }
            (candidates_dir / "episode-001.json").write_text(
                json.dumps(candidate_doc), encoding="utf-8"
            )
            shortlist = {
                "schema_version": "1.0",
                "proposals": [proposal, second_proposal],
            }
            proposal_file = root / "shortlist.json"
            proposal_file.write_text(json.dumps(shortlist), encoding="utf-8")
            config = LLMConfig(
                provider="gemini",
                model="gemini-3.5-flash-lite",
                api_key=None,
                batch_size=1,
                max_retries=0,
                timeout_seconds=1,
                free_tier_confirmed=False,
            )
            dry_run = run_proposal_evaluations(
                proposal_file=proposal_file,
                parsed_directory=parsed_dir,
                candidate_directory=candidates_dir,
                output_directory=output_dir,
                config=config,
                dry_run=True,
            )
            self.assertEqual(dry_run["proposals_compatible"], 2)
            self.assertEqual(dry_run["api_calls"], 0)

            class FakeClient:
                calls = 0
                raw_responses = []

                def evaluate_proposal(self, proposal):
                    self.calls += 1
                    if self.calls == 2:
                        raise LLMQuotaExceededError("mock quota exhaustion")
                    raw_text = json.dumps(_evaluation(proposal))
                    self.raw_responses.append(raw_text)
                    return raw_text, {
                        "provider": "gemini",
                        "requested_model": config.model,
                        "returned_model": config.model,
                    }

            client = FakeClient()
            with patch("src.evaluation.promo_evaluator.create_provider", return_value=client):
                first = run_proposal_evaluations(
                    proposal_file=proposal_file,
                    parsed_directory=parsed_dir,
                    candidate_directory=candidates_dir,
                    output_directory=output_dir,
                    config=config,
                )
                second = run_proposal_evaluations(
                    proposal_file=proposal_file,
                    parsed_directory=parsed_dir,
                    candidate_directory=candidates_dir,
                    output_directory=output_dir,
                    config=config,
                )
            self.assertEqual(first["validated"], 1)
            self.assertTrue(first["stopped_for_quota"])
            self.assertEqual(second["validated"], 2)
            self.assertEqual(second["skipped_resumed"], 1)
            self.assertFalse(second["stopped_for_quota"])
            self.assertEqual(client.calls, 3)
            self.assertTrue(
                (output_dir / "validated" / "promo-test.json").is_file()
            )
            self.assertTrue(
                (output_dir / "raw_responses" / "promo-test.json").is_file()
            )
            raw_record = json.loads(
                (output_dir / "raw_responses" / "promo-test.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                raw_record["raw_response"],
                client.raw_responses[0],
            )

    def test_loader_checks_source_references_against_parsed_files(self) -> None:
        proposal, _, source_candidates = _fixture("SINGLE_SEGMENT")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            parsed_dir = root / "parsed"
            candidates_dir = root / "candidates"
            parsed_dir.mkdir()
            candidates_dir.mkdir()
            (parsed_dir / "episode-001.json").write_text(
                json.dumps(
                    {
                        "episode_id": "episode-001",
                        "episode_number": 1,
                        "scenes": [
                            {
                                "id": "episode-001:scene-001",
                                "events": [
                                    item["event"]
                                    for item in source_candidates[
                                        "episode-001:clip-0001"
                                    ]["events"]
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (candidates_dir / "episode-001.json").write_text(
                json.dumps({"candidates": list(source_candidates.values())}),
                encoding="utf-8",
            )
            proposal_file = root / "shortlist.json"
            proposal_file.write_text(
                json.dumps({"proposals": [proposal]}), encoding="utf-8"
            )
            loaded = load_proposals(
                proposal_file=proposal_file,
                parsed_directory=parsed_dir,
                candidate_directory=candidates_dir,
            )
            self.assertEqual(len(loaded), 1)
            tampered = json.loads(proposal_file.read_text(encoding="utf-8"))
            tampered["proposals"][0]["segments"][0]["events"][0]["event"]["content"][
                "raw"
            ] = "altered content"
            proposal_file.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationValidationError, "differs from parsed source"):
                load_proposals(
                    proposal_file=proposal_file,
                    parsed_directory=parsed_dir,
                    candidate_directory=candidates_dir,
                )

    def test_gemini_proposal_request_uses_segment_aware_schema(self) -> None:
        proposal = _normalized("CROSS_EPISODE_MULTI_SEGMENT")
        response_body = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "{}"}]},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash-lite",
        }
        captured: dict[str, object] = {}
        class FakeResponse:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *exception_details):
                return bool(exception_details[0]) and False

            def read(self):
                return json.dumps(self.body).encode("utf-8")

        def fake_open(request, timeout):
            payload = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            if request.full_url.endswith(":countTokens"):
                captured["count_payload"] = payload
                return FakeResponse({"totalTokens": 10})
            captured["url"] = request.full_url
            captured["payload"] = payload
            return FakeResponse(response_body)

        config = LLMConfig(
            provider="gemini",
            model="gemini-3.5-flash-lite",
            api_key="unit-test-key",
            batch_size=1,
            max_retries=0,
            timeout_seconds=2,
            free_tier_confirmed=True,
        )
        provider = GeminiProvider(config, opener=fake_open)
        provider.evaluate_proposal(proposal)
        payload = captured["payload"]
        self.assertIn("promo_id", payload["generationConfig"]["responseJsonSchema"]["properties"])
        self.assertIn(
            "playback_timestamp",
            payload["generationConfig"]["responseJsonSchema"]["properties"]["hook"][
                "properties"
            ]["evidence"]["items"]["properties"],
        )
        self.assertIn(
            "not original source chronology",
            payload["systemInstruction"]["parts"][0]["text"],
        )
        self.assertEqual(
            gemini_proposal_response_schema(),
            payload["generationConfig"]["responseJsonSchema"],
        )


if __name__ == "__main__":
    unittest.main()
