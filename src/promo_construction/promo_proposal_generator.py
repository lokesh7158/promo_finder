"""Build deterministic, source-traceable promo proposals from existing clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


MIN_PROMO_HUNDREDTHS = 2_000
MAX_PROMO_HUNDREDTHS = 6_000
MIN_FRAGMENT_HUNDREDTHS = 800
MAX_FRAGMENT_HUNDREDTHS = 2_200
SHORTLIST_TARGET = 200
SHORTLIST_CATEGORY_TARGETS = {
    "SINGLE_SEGMENT": 60,
    "SAME_EPISODE_MULTI_SEGMENT": 70,
    "CROSS_EPISODE_MULTI_SEGMENT": 70,
}
SHORTLIST_CATEGORY_DURATION_TARGETS = {
    "SINGLE_SEGMENT": {"45-60": 60},
    "SAME_EPISODE_MULTI_SEGMENT": {"30-<45": 70},
    "CROSS_EPISODE_MULTI_SEGMENT": {"20-<30": 50, "30-<45": 20},
}
SCHEMA_VERSION = "1.0"
TIMESTAMP_PATTERN = re.compile(r"^(\d{2}):(\d{2})\.(\d{2})$")
DRAMATIC_EVENT_TYPES = {
    "ACTION",
    "CHARACTER_DIALOGUE",
    "SOUND_FX",
    "VOICE_OVER",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _timestamp_hundredths(raw: Any) -> int:
    if not isinstance(raw, str):
        raise ValueError(f"Timestamp must be a string, got {raw!r}")
    match = TIMESTAMP_PATTERN.fullmatch(raw)
    if match is None or int(match.group(2)) >= 60:
        raise ValueError(f"Invalid source timestamp: {raw!r}")
    return int(match.group(1)) * 6_000 + int(match.group(2)) * 100 + int(match.group(3))


def _format_timestamp(hundredths: int) -> str:
    minutes, remainder = divmod(hundredths, 6_000)
    seconds, fraction = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{fraction:02d}"


def _source_event_ref(episode_id: str, event_id: str) -> str:
    if event_id.startswith(f"{episode_id}:"):
        return event_id
    return f"{episode_id}:{event_id}"


def _speaker_key(event: dict[str, Any]) -> str | None:
    raw = event.get("speaker_label_raw")
    if not isinstance(raw, str) or not raw.strip():
        return None
    label = re.sub(r"\([^)]*\)", "", raw).strip().upper()
    label = re.sub(r"\b(?:O\.S\.|O\.C\.|V\.O\.)\b", "", label).strip()
    label = re.sub(r"\s+", " ", label)
    return label or None


def _load_sources(
    parsed_directory: Path,
    candidate_directory: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    episodes: dict[str, dict[str, Any]] = {}
    source_events: dict[str, dict[str, Any]] = {}
    for path in sorted(parsed_directory.glob("*.json")):
        episode = _read_json(path)
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in episodes:
            raise ValueError(f"Invalid or duplicate episode ID in {path}")
        metadata = episode.get("metadata", {})
        number_value = metadata.get("episode_number", {}).get("value")
        if not isinstance(number_value, str):
            raise ValueError(f"Episode number missing in {path}")
        number_match = re.search(r"\d+", number_value)
        if number_match is None:
            raise ValueError(f"Episode number malformed in {path}: {number_value!r}")
        episode_number = int(number_match.group())
        episodes[episode_id] = {
            "episode_id": episode_id,
            "episode_number": episode_number,
            "logline": metadata.get("episode_logline", {}).get("value", ""),
            "continuity_notes": metadata.get("continuity_notes", {}).get("value", ""),
            "source_path": str(path),
            "source_episode_file": path.name,
        }
        for scene_index, scene in enumerate(episode.get("scenes", [])):
            scene_id = scene.get("id")
            for event_index, event in enumerate(scene.get("events", [])):
                event_id = event.get("id")
                if not isinstance(episode_id, str) or not isinstance(event_id, str):
                    raise ValueError(f"Invalid source event ID: {event_id!r}")
                event_ref = _source_event_ref(episode_id, event_id)
                if event_ref in source_events:
                    raise ValueError(f"Invalid or duplicate source event ID: {event_id!r}")
                source_events[event_ref] = {
                    "source_event_ref": event_ref,
                    "source_event_id": event_id,
                    "episode_id": episode_id,
                    "episode_number": episode_number,
                    "scene_id": scene_id,
                    "scene_index": scene_index,
                    "event_index": event_index,
                    "event": event,
                    "scene": scene,
                    "start": _timestamp_hundredths(event.get("start", {}).get("raw")),
                    "end": _timestamp_hundredths(event.get("end", {}).get("raw")),
                }

    candidates: list[dict[str, Any]] = []
    for path in sorted(candidate_directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        document = _read_json(path)
        records = document.get("candidates")
        if not isinstance(records, list):
            raise ValueError(f"Candidate list missing in {path}")
        candidates.extend(record for record in records if isinstance(record, dict))
    if not episodes or not candidates:
        raise ValueError("Parsed episodes or generated candidates are missing")
    return episodes, candidates, source_events


def _source_segment(
    *,
    episode: dict[str, Any],
    candidates_for_episode: list[dict[str, Any]],
    events_for_episode: dict[str, dict[str, Any]],
    start: int,
    end: int,
) -> dict[str, Any]:
    if start >= end:
        raise ValueError("Segment start must precede segment end")
    if not MIN_FRAGMENT_HUNDREDTHS <= end - start <= MAX_PROMO_HUNDREDTHS:
        raise ValueError("Segment duration is outside supported bounds")
    event_ids = sorted(
        (
            event_id
            for event_id, record in events_for_episode.items()
            if record["start"] < end
            and record["end"] > start
        ),
        key=lambda event_id: (
            events_for_episode[event_id]["start"],
            events_for_episode[event_id]["end"],
            events_for_episode[event_id]["scene_index"],
            events_for_episode[event_id]["event_index"],
        ),
    )
    if not event_ids:
        raise ValueError("Segment contains no overlapping source events")
    event_set = {
        events_for_episode[event_ref]["source_event_id"] for event_ref in event_ids
    }
    source_candidate_ids = sorted(
        candidate["candidate_id"]
        for candidate in candidates_for_episode
        if event_set.issubset(
            {
                wrapped.get("event", {}).get("id")
                for wrapped in candidate.get("events", [])
                if isinstance(wrapped, dict)
            }
        )
    )
    scenes: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    wrapped_events: list[dict[str, Any]] = []
    for event_id in event_ids:
        record = events_for_episode[event_id]
        event = record["event"]
        scene_id = record["scene_id"]
        if isinstance(scene_id, str) and scene_id not in seen_scenes:
            scene = record["scene"]
            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_number": scene.get("number"),
                    "heading": scene.get("heading", {}).get("raw"),
                }
            )
            seen_scenes.add(scene_id)
        wrapped_events.append(
            {
                "scene_id": scene_id,
                "source_event_ref": record["source_event_ref"],
                "source_event_id": record["source_event_id"],
                "event": event,
            }
        )
    return {
        "episode_id": episode["episode_id"],
        "episode_number": episode["episode_number"],
        "source_episode_file": episode["source_episode_file"],
        "start_timestamp": _format_timestamp(start),
        "end_timestamp": _format_timestamp(end),
        "duration_seconds": (end - start) / 100,
        "source_candidate_ids": source_candidate_ids,
        "source_event_ids": event_ids,
        "scene_ids": [scene["scene_id"] for scene in scenes],
        "scenes": scenes,
        "events": wrapped_events,
    }


def _segment_signals(segment: dict[str, Any]) -> dict[str, Any]:
    start = _timestamp_hundredths(segment["start_timestamp"])
    end = _timestamp_hundredths(segment["end_timestamp"])
    wrapped_events = segment["events"]
    events = [wrapped["event"] for wrapped in wrapped_events]
    speakers = sorted({key for event in events if (key := _speaker_key(event))})
    hook_events = [
        event
        for event in events
        if event.get("type") in DRAMATIC_EVENT_TYPES
        and _timestamp_hundredths(event.get("start", {}).get("raw")) < start + 300
        and _timestamp_hundredths(event.get("end", {}).get("raw")) > start
    ]
    cliff_events = [
        event
        for event in events
        if event.get("type") in DRAMATIC_EVENT_TYPES
        and _timestamp_hundredths(event.get("start", {}).get("raw")) < end
        and _timestamp_hundredths(event.get("end", {}).get("raw")) > max(start, end - 300)
    ]
    counts = Counter(event.get("type", "UNKNOWN") for event in events)
    return {
        "event_count": len(events),
        "scene_count": len(segment["scenes"]),
        "event_type_counts": dict(sorted(counts.items())),
        "explicit_speaker_labels": speakers,
        "hook_opportunity": bool(hook_events),
        "cliffhanger_opportunity": bool(cliff_events),
        "hook_opportunity_event_ids": [
            wrapped["source_event_ref"]
            for wrapped in wrapped_events
            if wrapped["event"] in hook_events
        ],
        "cliffhanger_opportunity_event_ids": [
            wrapped["source_event_ref"]
            for wrapped in wrapped_events
            if wrapped["event"] in cliff_events
        ],
    }


def _proposal_id(category: str, segments: list[dict[str, Any]]) -> str:
    key = "|".join(
        f"{segment['episode_id']}:{segment['start_timestamp']}-{segment['end_timestamp']}"
        for segment in segments
    )
    digest = hashlib.sha256(f"{category}|{key}".encode("utf-8")).hexdigest()[:12]
    return f"promo-{category.lower()}-{digest}"


def _build_proposal(
    category: str,
    segments: list[dict[str, Any]],
    *,
    rationale: str,
    generation_method: str,
    source_candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    ordered_segments = []
    total_hundredths = 0
    all_event_ids: list[str] = []
    for order, segment in enumerate(segments, start=1):
        current = dict(segment)
        current["segment_order"] = order
        current["selection_signals"] = _segment_signals(current)
        current["playback_offset_seconds"] = total_hundredths / 100
        total_hundredths += round(current["duration_seconds"] * 100)
        current["playback_end_seconds"] = total_hundredths / 100
        all_event_ids.extend(current["source_event_ids"])
        ordered_segments.append(current)
    seconds = total_hundredths / 100
    promo_id = _proposal_id(category, ordered_segments)
    for segment in ordered_segments:
        segment["segment_id"] = f"{promo_id}:segment-{segment['segment_order']:02d}"
    candidate_ids = set(source_candidate_ids or [])
    for segment in ordered_segments:
        candidate_ids.update(segment["source_candidate_ids"])
    return {
        "promo_id": promo_id,
        "promo_category": category,
        "segments": ordered_segments,
        "total_duration_seconds": seconds,
        "source_candidate_ids": sorted(candidate_ids),
        "source_event_ids": list(dict.fromkeys(all_event_ids)),
        "segment_order": [segment["segment_order"] for segment in ordered_segments],
        "evaluation_handoff": {
            "format": "multi_segment_source_timeline_v1",
            "playback_offsets_are_derived": True,
            "source_timestamps_are_unchanged": True,
            "requires_segment_aware_evaluation_schema": len(ordered_segments) > 1,
        },
        "narrative_rationale": rationale,
        "candidate_generation_method": generation_method,
        "validation_status": "PENDING",
        "rejection_reasons": [],
        "evaluation_note": (
            "Hook and ending fields are structural opportunities only, not a judgment "
            "that the promo is effective."
        ),
    }


def _fragment_candidates(
    episodes: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    source_events: dict[str, dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    candidates_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_episode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        candidates_by_episode[candidate.get("episode", {}).get("episode_id")].append(candidate)
    for event_id, record in source_events.items():
        events_by_episode[record["episode_id"]][event_id] = record

    raw_fragments: dict[tuple[str, int, int], dict[str, Any]] = {}
    for candidate in candidates:
        episode_info = candidate.get("episode", {})
        episode = episodes.get(episode_info.get("episode_id"))
        if episode is None:
            continue
        try:
            candidate_start = _timestamp_hundredths(candidate["time_range"]["start"]["raw"])
            candidate_end = _timestamp_hundredths(candidate["time_range"]["end"]["raw"])
        except (KeyError, TypeError, ValueError):
            continue
        event_records = []
        boundary_values = {candidate_start, candidate_end}
        for wrapped in candidate.get("events", []):
            raw_event_id = wrapped.get("event", {}).get("id")
            event_ref = (
                _source_event_ref(episode["episode_id"], raw_event_id)
                if isinstance(raw_event_id, str)
                else None
            )
            record = source_events.get(event_ref)
            if record is None:
                continue
            event_records.append(record)
            if candidate_start <= record["start"] <= candidate_end:
                boundary_values.add(record["start"])
            if candidate_start <= record["end"] <= candidate_end:
                boundary_values.add(record["end"])
        boundaries = sorted(boundary_values)
        for left_index, start in enumerate(boundaries):
            for end in boundaries[left_index + 1 :]:
                duration = end - start
                if duration < MIN_FRAGMENT_HUNDREDTHS:
                    continue
                if duration > MAX_FRAGMENT_HUNDREDTHS:
                    break
                selected_events = [
                    record
                    for record in event_records
                    if record["start"] < end and record["end"] > start
                ]
                if not selected_events:
                    continue
                key = (episode["episode_id"], start, end)
                existing = raw_fragments.get(key)
                if existing is None:
                    segment = _source_segment(
                        episode=episode,
                        candidates_for_episode=candidates_by_episode[episode["episode_id"]],
                        events_for_episode=events_by_episode[episode["episode_id"]],
                        start=start,
                        end=end,
                    )
                    signals = _segment_signals(segment)
                    if not signals["explicit_speaker_labels"]:
                        continue
                    score = (
                        len(signals["explicit_speaker_labels"]) * 5
                        + signals["event_type_counts"].get("CHARACTER_DIALOGUE", 0) * 3
                        + signals["event_type_counts"].get("ACTION", 0) * 2
                        + int(signals["hook_opportunity"])
                        + int(signals["cliffhanger_opportunity"])
                    )
                    raw_fragments[key] = {
                        "segment": segment,
                        "signals": signals,
                        "score": score,
                        "speaker_set": set(signals["explicit_speaker_labels"]),
                        "midpoint": (start + end) / 2,
                    }
                elif candidate["candidate_id"] not in existing["segment"]["source_candidate_ids"]:
                    existing["segment"]["source_candidate_ids"].append(candidate["candidate_id"])

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fragment in raw_fragments.values():
        by_episode[fragment["segment"]["episode_number"]].append(fragment)

    selected_by_episode: dict[int, list[dict[str, Any]]] = {}
    for episode_number, fragments in by_episode.items():
        episode_end = max(
            _timestamp_hundredths(fragment["segment"]["end_timestamp"])
            for fragment in fragments
        )
        ordered = sorted(
            fragments,
            key=lambda item: (-item["score"], item["segment"]["start_timestamp"]),
        )
        selected: list[dict[str, Any]] = []
        speaker_coverage: set[str] = set()
        region_coverage: set[int] = set()
        # Favor separate timeline portions and speaker coverage, then fill a small cap.
        for fragment in ordered:
            start = _timestamp_hundredths(fragment["segment"]["start_timestamp"])
            region = min(4, start * 5 // max(episode_end, 1))
            new_speaker = not fragment["speaker_set"].issubset(speaker_coverage)
            if region not in region_coverage or new_speaker:
                selected.append(fragment)
                speaker_coverage.update(fragment["speaker_set"])
                region_coverage.add(region)
                if len(selected) >= 8:
                    break
        if len(selected) < 8:
            seen = {
                (
                    item["segment"]["start_timestamp"],
                    item["segment"]["end_timestamp"],
                )
                for item in selected
            }
            for fragment in ordered:
                signature = (
                    fragment["segment"]["start_timestamp"],
                    fragment["segment"]["end_timestamp"],
                )
                if signature not in seen:
                    selected.append(fragment)
                    seen.add(signature)
                if len(selected) >= 8:
                    break
        selected_by_episode[episode_number] = selected
    return selected_by_episode


def _segment_has_hook(segment: dict[str, Any]) -> bool:
    return _segment_signals(segment)["hook_opportunity"]


def _segment_has_cliffhanger(segment: dict[str, Any]) -> bool:
    return _segment_signals(segment)["cliffhanger_opportunity"]


def _validate_proposal(
    proposal: dict[str, Any],
    episodes: dict[str, dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
    source_events: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    segments = proposal.get("segments")
    if not isinstance(segments, list) or not segments:
        return ["proposal_has_no_segments"]
    total = 0
    segment_keys: set[tuple[str, str, str]] = set()
    proposal_events: set[str] = set()
    for index, segment in enumerate(segments):
        episode = episodes.get(segment.get("episode_id"))
        if episode is None or episode["episode_number"] != segment.get("episode_number"):
            errors.append(f"segment_{index + 1}_unknown_or_mismatched_episode")
            continue
        try:
            start = _timestamp_hundredths(segment.get("start_timestamp"))
            end = _timestamp_hundredths(segment.get("end_timestamp"))
        except ValueError:
            errors.append(f"segment_{index + 1}_invalid_timestamp")
            continue
        if start >= end:
            errors.append(f"segment_{index + 1}_invalid_boundaries")
            continue
        if end - start != round(segment.get("duration_seconds", -1) * 100):
            errors.append(f"segment_{index + 1}_duration_mismatch")
        if not MIN_FRAGMENT_HUNDREDTHS <= end - start <= MAX_PROMO_HUNDREDTHS:
            errors.append(f"segment_{index + 1}_duration_out_of_range")
        segment_key = (segment["episode_id"], segment["start_timestamp"], segment["end_timestamp"])
        if segment_key in segment_keys:
            errors.append("duplicate_segment")
        segment_keys.add(segment_key)
        total += end - start
        referenced_events = set(segment.get("source_event_ids", []))
        if not referenced_events:
            errors.append(f"segment_{index + 1}_has_no_source_events")
        for event_id in referenced_events:
            record = source_events.get(event_id)
            if record is None or record["episode_id"] != segment["episode_id"]:
                errors.append(f"segment_{index + 1}_unknown_event:{event_id}")
                continue
            if not (record["start"] < end and record["end"] > start):
                errors.append(f"segment_{index + 1}_event_does_not_overlap:{event_id}")
            proposal_events.add(event_id)
        for candidate_id in segment.get("source_candidate_ids", []):
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None or candidate.get("episode", {}).get("episode_id") != segment["episode_id"]:
                errors.append(f"segment_{index + 1}_unknown_source_candidate:{candidate_id}")
        if segment.get("segment_order") != index + 1:
            errors.append(f"segment_{index + 1}_order_mismatch")
        if index:
            previous = segments[index - 1]
            if proposal["promo_category"] == "SAME_EPISODE_MULTI_SEGMENT":
                previous_end = _timestamp_hundredths(previous["end_timestamp"])
                current_start = _timestamp_hundredths(segment["start_timestamp"])
                if previous["episode_id"] != segment["episode_id"] or previous_end > current_start:
                    errors.append("same_episode_segments_overlap_or_are_out_of_order")
            elif proposal["promo_category"] == "CROSS_EPISODE_MULTI_SEGMENT":
                if segment["episode_number"] <= previous["episode_number"]:
                    errors.append("cross_episode_segments_not_in_episode_order")
    if not MIN_PROMO_HUNDREDTHS <= total <= MAX_PROMO_HUNDREDTHS:
        errors.append("assembled_duration_out_of_range")
    if len(proposal_events) != len(proposal.get("source_event_ids", [])):
        if set(proposal.get("source_event_ids", [])) != proposal_events:
            errors.append("proposal_event_traceability_mismatch")
    if not proposal.get("narrative_rationale", "").strip():
        errors.append("missing_narrative_rationale")
    if len(segments) > 1 and not proposal.get("narrative_rationale"):
        errors.append("multi_segment_missing_narrative_relationship")
    if segments and not _segment_has_hook(segments[0]):
        errors.append("no_plausible_hook_opportunity_in_opening_window")
    if segments and not _segment_has_cliffhanger(segments[-1]):
        errors.append("no_plausible_unresolved_ending_opportunity")
    return list(dict.fromkeys(errors))


def _proposal_quality(proposal: dict[str, Any]) -> tuple[int, float, str]:
    signals = [segment["selection_signals"] for segment in proposal["segments"]]
    meaningful = sum(
        signal["event_type_counts"].get("CHARACTER_DIALOGUE", 0) * 3
        + signal["event_type_counts"].get("ACTION", 0) * 2
        + signal["event_count"]
        for signal in signals
    )
    coverage = len(set(proposal["source_event_ids"]))
    return meaningful + coverage, proposal["total_duration_seconds"], proposal["promo_id"]


def _duration_band(seconds: float) -> str:
    if seconds < 30:
        return "20-<30"
    if seconds < 45:
        return "30-<45"
    return "45-60"


def _proposal_similarity(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["promo_category"] != second["promo_category"]:
        return False
    if len(first["segments"]) != len(second["segments"]):
        return False
    first_events = set(first["source_event_ids"])
    second_events = set(second["source_event_ids"])
    event_union = first_events | second_events
    event_similarity = len(first_events & second_events) / len(event_union) if event_union else 1.0
    if event_similarity >= 0.75:
        return True
    for left, right in zip(first["segments"], second["segments"]):
        if left["episode_id"] != right["episode_id"]:
            return False
        left_start = _timestamp_hundredths(left["start_timestamp"])
        left_end = _timestamp_hundredths(left["end_timestamp"])
        right_start = _timestamp_hundredths(right["start_timestamp"])
        right_end = _timestamp_hundredths(right["end_timestamp"])
        intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
        union = max(left_end, right_end) - min(left_start, right_start)
        interval_iou = intersection / union if union else 1.0
        left_events = set(left["source_event_ids"])
        right_events = set(right["source_event_ids"])
        event_union = left_events | right_events
        event_iou = len(left_events & right_events) / len(event_union) if event_union else 1.0
        if interval_iou < 0.85 or event_iou < 0.80:
            return False
    return True


def _high_overlap_pairs(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for left_index, first in enumerate(proposals):
        first_events = set(first["source_event_ids"])
        for second in proposals[left_index + 1 :]:
            second_events = set(second["source_event_ids"])
            union = first_events | second_events
            similarity = len(first_events & second_events) / len(union) if union else 0.0
            if similarity >= 0.75:
                ranked.append(
                    {
                        "promo_id_a": first["promo_id"],
                        "promo_id_b": second["promo_id"],
                        "category_a": first["promo_category"],
                        "category_b": second["promo_category"],
                        "source_event_jaccard": round(similarity, 4),
                    }
                )
    ranked.sort(key=lambda item: (-item["source_event_jaccard"], item["promo_id_a"], item["promo_id_b"]))
    return ranked[:20]


def _diverse_shortlist(
    proposals: list[dict[str, Any]],
    *,
    target: int = SHORTLIST_TARGET,
) -> tuple[list[dict[str, Any]], int]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        if proposal["validation_status"] == "VALID":
            by_category[proposal["promo_category"]].append(proposal)
    for records in by_category.values():
        records.sort(key=_proposal_quality, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    episode_usage: Counter[int] = Counter()
    category_usage: Counter[str] = Counter()
    category_duration_usage: Counter[tuple[str, str]] = Counter()
    max_category = max(
        max(SHORTLIST_CATEGORY_TARGETS.values()),
        target // 2,
    )
    duplicates_suppressed = 0
    category_order = list(SHORTLIST_CATEGORY_TARGETS)
    while len(selected) < target:
        made_progress = False
        for category in category_order:
            if len(selected) >= target:
                break
            desired = SHORTLIST_CATEGORY_TARGETS[category]
            if category_usage[category] >= desired or category_usage[category] >= max_category:
                continue
            preferred_bands = SHORTLIST_CATEGORY_DURATION_TARGETS[category]
            for band, band_target in preferred_bands.items():
                if category_duration_usage[(category, band)] >= band_target:
                    continue
                for proposal in by_category.get(category, []):
                    if proposal["promo_id"] in selected_ids:
                        continue
                    if _duration_band(proposal["total_duration_seconds"]) != band:
                        continue
                    involved_episodes = {
                        segment["episode_number"] for segment in proposal["segments"]
                    }
                    per_episode_limit = 3 if category == "SINGLE_SEGMENT" else 6
                    if any(
                        episode_usage[number] >= per_episode_limit
                        for number in involved_episodes
                    ):
                        continue
                    if any(_proposal_similarity(proposal, prior) for prior in selected):
                        duplicates_suppressed += 1
                        selected_ids.add(proposal["promo_id"])
                        continue
                    proposal["shortlist_rank"] = len(selected) + 1
                    selected.append(proposal)
                    selected_ids.add(proposal["promo_id"])
                    category_usage[category] += 1
                    category_duration_usage[(category, band)] += 1
                    for number in involved_episodes:
                        episode_usage[number] += 1
                    made_progress = True
                    break
                if made_progress:
                    break
        if not made_progress:
            break
    while len(selected) < target:
        made_progress = False
        for category in category_order:
            if len(selected) >= target:
                break
            if category_usage[category] >= max_category:
                continue
            for proposal in by_category.get(category, []):
                if proposal["promo_id"] in selected_ids:
                    continue
                band = _duration_band(proposal["total_duration_seconds"])
                involved_episodes = {
                    segment["episode_number"] for segment in proposal["segments"]
                }
                per_episode_limit = 3 if category == "SINGLE_SEGMENT" else 6
                if any(episode_usage[number] >= per_episode_limit for number in involved_episodes):
                    continue
                if any(_proposal_similarity(proposal, prior) for prior in selected):
                    duplicates_suppressed += 1
                    selected_ids.add(proposal["promo_id"])
                    continue
                proposal["shortlist_rank"] = len(selected) + 1
                selected.append(proposal)
                selected_ids.add(proposal["promo_id"])
                category_usage[category] += 1
                category_duration_usage[(category, band)] += 1
                for number in involved_episodes:
                    episode_usage[number] += 1
                made_progress = True
                break
        if not made_progress:
            break
    return selected, duplicates_suppressed


def generate_proposals(
    parsed_directory: Path,
    candidate_directory: Path,
) -> dict[str, Any]:
    """Generate, validate, and diversity-filter complete promo proposals."""
    episodes, candidates, source_events = _load_sources(parsed_directory, candidate_directory)
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if len(candidates_by_id) != len(candidates):
        raise ValueError("Duplicate candidate IDs found across candidate files")
    candidates_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_episode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        candidates_by_episode[candidate.get("episode", {}).get("episode_id")].append(candidate)
    for event_id, record in source_events.items():
        events_by_episode[record["episode_id"]][event_id] = record

    proposals: list[dict[str, Any]] = []

    for candidate in candidates:
        episode = episodes[candidate["episode"]["episode_id"]]
        start = _timestamp_hundredths(candidate["time_range"]["start"]["raw"])
        end = _timestamp_hundredths(candidate["time_range"]["end"]["raw"])
        segment = _source_segment(
            episode=episode,
            candidates_for_episode=candidates_by_episode[episode["episode_id"]],
            events_for_episode=events_by_episode[episode["episode_id"]],
            start=start,
            end=end,
        )
        proposals.append(
            _build_proposal(
                "SINGLE_SEGMENT",
                [segment],
                rationale=(
                    "A single existing continuous candidate is retained as a complete "
                    "promo opportunity; hook and ending suitability remain for evaluation."
                ),
                generation_method="reuse_existing_candidate",
                source_candidate_ids=[candidate["candidate_id"]],
            )
        )

    fragments_by_episode = _fragment_candidates(episodes, candidates, source_events)
    same_episode_proposals: list[dict[str, Any]] = []
    for episode_number, fragments in sorted(fragments_by_episode.items()):
        for first in fragments:
            first_segment = first["segment"]
            if not first["signals"]["hook_opportunity"]:
                continue
            first_end = _timestamp_hundredths(first_segment["end_timestamp"])
            for second in fragments:
                second_segment = second["segment"]
                second_start = _timestamp_hundredths(second_segment["start_timestamp"])
                if (
                    first_segment["start_timestamp"] == second_segment["start_timestamp"]
                    and first_segment["end_timestamp"] == second_segment["end_timestamp"]
                ) or first_end > second_start:
                    continue
                shared_speakers = sorted(first["speaker_set"] & second["speaker_set"])
                if not shared_speakers or not second["signals"]["cliffhanger_opportunity"]:
                    continue
                duration = first_segment["duration_seconds"] + second_segment["duration_seconds"]
                if not 20 <= duration <= 60:
                    continue
                same_episode_proposals.append(
                    _build_proposal(
                        "SAME_EPISODE_MULTI_SEGMENT",
                        [first_segment, second_segment],
                        rationale=(
                            "Source moments occur in episode order and share explicit "
                            f"speaker label(s): {', '.join(shared_speakers)}. "
                            "Their plot connection is a hypothesis for human/LLM review."
                        ),
                        generation_method="shared_speaker_chronological_pair",
                    )
                )

    cross_episode_proposals: list[dict[str, Any]] = []
    by_speaker_episode: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode_number, fragments in sorted(fragments_by_episode.items()):
        for fragment in fragments:
            for speaker in fragment["speaker_set"]:
                by_speaker_episode[speaker][episode_number].append(fragment)
    for speaker, episode_map in sorted(by_speaker_episode.items()):
        episode_numbers = sorted(episode_map)
        for first_pos, first_episode in enumerate(episode_numbers):
            for second_episode in episode_numbers[first_pos + 1 :]:
                if second_episode - first_episode > 5:
                    break
                for first in episode_map[first_episode]:
                    if not first["signals"]["hook_opportunity"]:
                        continue
                    for second in episode_map[second_episode]:
                        if not second["signals"]["cliffhanger_opportunity"]:
                            continue
                        duration = (
                            first["segment"]["duration_seconds"]
                            + second["segment"]["duration_seconds"]
                        )
                        if not 20 <= duration <= 60:
                            continue
                        cross_episode_proposals.append(
                            _build_proposal(
                                "CROSS_EPISODE_MULTI_SEGMENT",
                                [first["segment"], second["segment"]],
                                rationale=(
                                    "The source segments share the exact normalized "
                                    f"speaker label {speaker!r} and follow episode-number "
                                    "order. Continuity of the specific conflict is not "
                                    "assumed and requires human/LLM review."
                                ),
                                generation_method="shared_speaker_episode_order_pair",
                            )
                        )

    unique: dict[str, dict[str, Any]] = {}
    for proposal in proposals + same_episode_proposals + cross_episode_proposals:
        unique.setdefault(proposal["promo_id"], proposal)
    proposals = list(unique.values())
    rejected: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for proposal in proposals:
        errors = _validate_proposal(proposal, episodes, candidates_by_id, source_events)
        if errors:
            proposal["validation_status"] = "REJECTED"
            proposal["rejection_reasons"] = errors
            rejected.append(proposal)
        else:
            proposal["validation_status"] = "VALID"
            valid.append(proposal)

    shortlist, duplicate_count = _diverse_shortlist(valid)
    final_ids = {proposal["promo_id"] for proposal in shortlist}
    not_shortlisted = [
        proposal
        for proposal in valid
        if proposal["promo_id"] not in final_ids
    ]
    duration_distribution: Counter[str] = Counter()
    for proposal in shortlist:
        duration = proposal["total_duration_seconds"]
        band = "20-<30" if duration < 30 else "30-<45" if duration < 45 else "45-60"
        duration_distribution[band] += 1
    episode_coverage = sorted(
        {
            segment["episode_number"]
            for proposal in shortlist
            for segment in proposal["segments"]
        }
    )
    scene_coverage = sorted(
        {
            scene_id
            for proposal in shortlist
            for segment in proposal["segments"]
            for scene_id in segment["scene_ids"]
        }
    )
    overlap_pairs = _high_overlap_pairs(shortlist)
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_configuration": {
            "minimum_total_duration_seconds": MIN_PROMO_HUNDREDTHS / 100,
            "maximum_total_duration_seconds": MAX_PROMO_HUNDREDTHS / 100,
            "minimum_multi_segment_part_seconds": MIN_FRAGMENT_HUNDREDTHS / 100,
            "maximum_segment_seconds": MAX_FRAGMENT_HUNDREDTHS / 100,
            "shortlist_target": SHORTLIST_TARGET,
            "category_targets": SHORTLIST_CATEGORY_TARGETS,
            "category_duration_targets": SHORTLIST_CATEGORY_DURATION_TARGETS,
            "cross_episode_max_gap": 5,
            "multi_segment_link_rule": (
                "exact normalized explicit speaker-label overlap; not proof of plot continuity"
            ),
            "hook_cliffhanger_prefilter": (
                "at least one overlapping source ACTION, CHARACTER_DIALOGUE, SOUND_FX, "
                "or VOICE_OVER event in the relevant 3-second window"
            ),
        },
        "summary": {
            "source_episode_count": len(episodes),
            "source_candidate_count": len(candidates),
            "source_event_count": len(source_events),
            "proposal_count_before_validation": len(proposals),
            "valid_proposal_count": len(valid),
            "rejected_proposal_count": len(rejected),
            "shortlist_count": len(shortlist),
            "shortlist_category_counts": dict(
                Counter(proposal["promo_category"] for proposal in shortlist)
            ),
            "rejection_reasons": dict(
                Counter(
                    reason
                    for proposal in rejected
                    for reason in proposal["rejection_reasons"]
                )
            ),
            "duplicate_proposals_suppressed": duplicate_count,
            "valid_not_shortlisted_count": len(not_shortlisted),
            "shortlist_episode_coverage_count": len(episode_coverage),
            "shortlist_episode_numbers": episode_coverage,
            "shortlist_scene_coverage_count": len(scene_coverage),
            "shortlist_duration_distribution": dict(duration_distribution),
            "shortlist_high_overlap_pairs": overlap_pairs,
            "notes": [
                "Timeline and speaker-label signals are deterministic structural filters, not promotional-quality judgments.",
                "Cross-episode shared labels suggest character continuity only; narrative coherence requires review.",
            ],
        },
        "all_proposals": proposals,
        "rejected_proposals": rejected,
        "shortlist": shortlist,
        "not_shortlisted": not_shortlisted,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_proposal(proposal: dict[str, Any], *, include_event_details: bool) -> dict[str, Any]:
    result = dict(proposal)
    segments = []
    for segment in proposal["segments"]:
        output_segment = dict(segment)
        if not include_event_details:
            output_segment.pop("events", None)
            output_segment["events_embedded"] = False
        else:
            output_segment["events_embedded"] = True
        segments.append(output_segment)
    result["segments"] = segments
    return result


def _write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    options = {"ensure_ascii": False}
    if compact:
        serialized = json.dumps(payload, separators=(",", ":"), **options)
    else:
        serialized = json.dumps(payload, indent=2, **options)
    path.write_text(serialized + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    parser.add_argument("--parsed-dir", type=Path, default=root / "data" / "output" / "parsed")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=root / "data" / "output" / "candidates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "output" / "promo_proposals",
    )
    arguments = parser.parse_args(argv)
    result = generate_proposals(arguments.parsed_dir, arguments.candidate_dir)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("all_proposals", "rejected_proposals", "shortlist", "not_shortlisted"):
        include_event_details = name == "shortlist"
        proposals = [
            _output_proposal(proposal, include_event_details=include_event_details)
            for proposal in result[name]
        ]
        if name == "not_shortlisted":
            proposals = [
                {
                    "promo_id": proposal["promo_id"],
                    "promo_category": proposal["promo_category"],
                    "segments": [
                        {
                            "segment_order": segment["segment_order"],
                            "episode_id": segment["episode_id"],
                            "start_timestamp": segment["start_timestamp"],
                            "end_timestamp": segment["end_timestamp"],
                        }
                        for segment in proposal["segments"]
                    ],
                    "shortlist_status": "not_selected",
                }
                for proposal in proposals
            ]
        payload = {"schema_version": SCHEMA_VERSION, "proposals": proposals}
        _write_json(
            arguments.output_dir / f"{name}.json",
            payload,
            compact=name != "shortlist",
        )
    _write_json(
        arguments.output_dir / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generation_configuration": result["generation_configuration"],
            "summary": result["summary"],
        },
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
