"""Generate deterministic, structurally diverse screenplay clip candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MIN_DURATION_HUNDREDTHS = 2_000
MAX_DURATION_HUNDREDTHS = 6_000
TIMELINE_REGION_COUNT = 5
SCHEMA_VERSION = "1.0"
TEMPORAL_IOU_THRESHOLD = 0.90
EVENT_JACCARD_THRESHOLD = 0.90
TIMESTAMP_PATTERN = re.compile(r"^(?P<minutes>\d{2}):(?P<seconds>\d{2})\.(?P<hundredths>\d{2})$")
DURATION_BANDS: tuple[tuple[str, int, int, int], ...] = (
    ("20-<30", 2_000, 3_000, 2_500),
    ("30-<45", 3_000, 4_500, 3_750),
    ("45-60", 4_500, 6_000, 5_250),
)


@dataclass(frozen=True)
class EventRecord:
    """Source event with its owning scene and integer timestamp boundaries."""

    scene_id: str
    scene_index: int
    event: dict[str, Any]
    start_hundredths: int
    end_hundredths: int


@dataclass(frozen=True)
class BoundaryReference:
    """An event timestamp that may be used as a candidate boundary."""

    event_id: str
    scene_id: str
    field: str
    raw: str


@dataclass
class CandidateWindow:
    """Lightweight Stage A window; full event data is attached only at selection."""

    start_hundredths: int
    end_hundredths: int
    timeline_region: int
    duration_band: str
    scene_ids: tuple[str, ...]
    event_indices: tuple[int, ...]
    event_ids: frozenset[str]
    selection_score: tuple[float, int, int, int]
    selection_reasons: list[str] = field(default_factory=list)


def _timestamp_to_hundredths(timestamp: dict[str, Any]) -> int | None:
    """Convert a parsed timestamp to exact integer hundredths."""
    raw = timestamp.get("raw")
    if not isinstance(raw, str):
        return None
    match = TIMESTAMP_PATTERN.fullmatch(raw)
    if match is None or int(match.group("seconds")) >= 60:
        return None
    return (
        int(match.group("minutes")) * 6_000
        + int(match.group("seconds")) * 100
        + int(match.group("hundredths"))
    )


def _duration_band(duration_hundredths: int) -> tuple[str, int] | None:
    """Return duration band and its midpoint, using inclusive 20–60s bounds."""
    for label, minimum, maximum, midpoint in DURATION_BANDS:
        upper_match = duration_hundredths < maximum
        if maximum == MAX_DURATION_HUNDREDTHS:
            upper_match = duration_hundredths <= maximum
        if minimum <= duration_hundredths and upper_match:
            return label, midpoint
    return None


def _episode_number(episode: dict[str, Any]) -> int | None:
    """Read the internal episode number without rewriting its raw metadata."""
    metadata = episode.get("metadata")
    if not isinstance(metadata, dict):
        return None
    episode_field = metadata.get("episode_number")
    if not isinstance(episode_field, dict):
        return None
    value = episode_field.get("value")
    if not isinstance(value, str):
        return None
    match = re.search(r"\bEpisode\s+(\d+)\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _load_episode(path: Path) -> dict[str, Any]:
    """Load one parsed episode JSON file."""
    with path.open("r", encoding="utf-8") as file:
        episode = json.load(file)
    if not isinstance(episode, dict) or not isinstance(episode.get("scenes"), list):
        raise ValueError(f"Invalid parsed episode structure in {path}")
    if not isinstance(episode.get("episode_id"), str):
        raise ValueError(f"Missing episode_id in {path}")
    return episode


def _collect_source_data(
    episode: dict[str, Any],
) -> tuple[
    list[EventRecord],
    list[tuple[str, int, int, dict[str, Any]]],
    dict[int, list[BoundaryReference]],
    int,
    int,
]:
    """Collect valid event and scene ranges plus event-backed boundary references."""
    event_records: list[EventRecord] = []
    scene_ranges: list[tuple[str, int, int, dict[str, Any]]] = []
    boundary_references: dict[int, list[BoundaryReference]] = {}
    invalid_event_timestamp_count = 0
    max_scene_end = 0

    for scene_index, scene in enumerate(episode["scenes"], start=1):
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get("id")
        if not isinstance(scene_id, str):
            scene_id = f"{episode['episode_id']}:scene-{scene_index:03d}"
        scene_start = _timestamp_to_hundredths(scene.get("start", {}))
        scene_end = _timestamp_to_hundredths(scene.get("end", {}))
        if scene_start is not None and scene_end is not None and scene_start <= scene_end:
            scene_ranges.append((scene_id, scene_start, scene_end, scene))
            max_scene_end = max(max_scene_end, scene_end)

        events = scene.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                invalid_event_timestamp_count += 1
                continue
            start = _timestamp_to_hundredths(event.get("start", {}))
            end = _timestamp_to_hundredths(event.get("end", {}))
            if start is None or end is None or start > end:
                invalid_event_timestamp_count += 1
                continue

            event_records.append(
                EventRecord(
                    scene_id=scene_id,
                    scene_index=scene_index,
                    event=event,
                    start_hundredths=start,
                    end_hundredths=end,
                )
            )
            event_id = str(event.get("id", ""))
            for field_name, timestamp, hundredths in (
                ("start", event.get("start", {}), start),
                ("end", event.get("end", {}), end),
            ):
                raw = timestamp.get("raw")
                if isinstance(raw, str):
                    boundary_references.setdefault(hundredths, []).append(
                        BoundaryReference(
                            event_id=event_id,
                            scene_id=scene_id,
                            field=field_name,
                            raw=raw,
                        )
                    )

    if max_scene_end == 0:
        max_scene_end = max(
            (record.end_hundredths for record in event_records),
            default=0,
        )
    return (
        event_records,
        scene_ranges,
        boundary_references,
        max_scene_end,
        invalid_event_timestamp_count,
    )


def _timeline_region(midpoint_hundredths: int, episode_duration: int) -> int:
    """Assign a candidate midpoint to one of five episode timeline regions."""
    if episode_duration <= 0:
        return 0
    return min(
        TIMELINE_REGION_COUNT - 1,
        midpoint_hundredths * TIMELINE_REGION_COUNT // episode_duration,
    )


def _window_score(
    start_hundredths: int,
    end_hundredths: int,
    region: int,
    episode_duration: int,
    band_midpoint: int,
) -> tuple[float, int, int, int]:
    """Rank candidates by distance to their region and duration-band centers."""
    region_center = episode_duration * (2 * region + 1) / (2 * TIMELINE_REGION_COUNT)
    midpoint = (start_hundredths + end_hundredths) / 2
    duration = end_hundredths - start_hundredths
    region_width = max(1, episode_duration / TIMELINE_REGION_COUNT)
    score = abs(midpoint - region_center) / region_width
    return score, abs(duration - band_midpoint), start_hundredths, end_hundredths


def _scene_ids_for_window(
    scene_ranges: list[tuple[str, int, int, dict[str, Any]]],
    start_hundredths: int,
    end_hundredths: int,
) -> tuple[str, ...]:
    """Return scenes intersecting a candidate using strict interval overlap."""
    return tuple(
        scene_id
        for scene_id, scene_start, scene_end, _ in scene_ranges
        if scene_start < end_hundredths and scene_end > start_hundredths
    )


def _candidate_window(
    start_hundredths: int,
    end_hundredths: int,
    *,
    episode_duration: int,
    event_records: list[EventRecord],
    scene_ranges: list[tuple[str, int, int, dict[str, Any]]],
) -> CandidateWindow | None:
    """Build a Stage A candidate summary and its strict-overlap event set."""
    band = _duration_band(end_hundredths - start_hundredths)
    if band is None:
        return None

    midpoint = (start_hundredths + end_hundredths) // 2
    region = _timeline_region(midpoint, episode_duration)
    event_indices = tuple(
        index
        for index, record in enumerate(event_records)
        if record.start_hundredths < end_hundredths
        and record.end_hundredths > start_hundredths
    )
    if not event_indices:
        return None
    scene_ids = _scene_ids_for_window(scene_ranges, start_hundredths, end_hundredths)
    return CandidateWindow(
        start_hundredths=start_hundredths,
        end_hundredths=end_hundredths,
        timeline_region=region,
        duration_band=band[0],
        scene_ids=scene_ids,
        event_indices=event_indices,
        event_ids=frozenset(
            str(event_records[index].event.get("id", "")) for index in event_indices
        ),
        selection_score=_window_score(
            start_hundredths,
            end_hundredths,
            region,
            episode_duration,
            band[1],
        ),
    )


def _window_key(window: CandidateWindow) -> tuple[int, int]:
    return window.start_hundredths, window.end_hundredths


def _temporal_iou(left: CandidateWindow, right: CandidateWindow) -> float:
    intersection = max(
        0,
        min(left.end_hundredths, right.end_hundredths)
        - max(left.start_hundredths, right.start_hundredths),
    )
    union = max(left.end_hundredths, right.end_hundredths) - min(
        left.start_hundredths, right.start_hundredths
    )
    return intersection / union if union else 1.0


def _event_jaccard(left: CandidateWindow, right: CandidateWindow) -> float:
    union = left.event_ids | right.event_ids
    if not union:
        return 1.0
    return len(left.event_ids & right.event_ids) / len(union)


def _is_near_duplicate(left: CandidateWindow, right: CandidateWindow) -> bool:
    return (
        left.timeline_region == right.timeline_region
        and left.duration_band == right.duration_band
        and left.scene_ids == right.scene_ids
        and _temporal_iou(left, right) >= TEMPORAL_IOU_THRESHOLD
        and _event_jaccard(left, right) >= EVENT_JACCARD_THRESHOLD
    )


def _select_stage_b(
    event_records: list[EventRecord],
    scene_ranges: list[tuple[str, int, int, dict[str, Any]]],
    boundary_values: list[int],
    episode_duration: int,
) -> tuple[int, dict[str, int], list[CandidateWindow], int, int]:
    """Stream Stage A windows and retain deterministic Stage B representatives."""
    cell_representatives: dict[tuple[int, str], CandidateWindow] = {}
    transition_representatives: dict[int, CandidateWindow] = {}
    stage_a_count = 0
    stage_a_duration_distribution = {
        label: 0 for label, _, _, _ in DURATION_BANDS
    }

    for start_index, start in enumerate(boundary_values):
        for end_index in range(start_index + 1, len(boundary_values)):
            end = boundary_values[end_index]
            duration = end - start
            if duration > MAX_DURATION_HUNDREDTHS:
                break
            if duration < MIN_DURATION_HUNDREDTHS:
                continue
            band = _duration_band(duration)
            if band is None:
                continue
            window = _candidate_window(
                start,
                end,
                episode_duration=episode_duration,
                event_records=event_records,
                scene_ranges=scene_ranges,
            )
            if window is None:
                continue
            stage_a_count += 1
            stage_a_duration_distribution[window.duration_band] += 1

            cell_key = window.timeline_region, window.duration_band
            current = cell_representatives.get(cell_key)
            if current is None or window.selection_score < current.selection_score:
                cell_representatives[cell_key] = window

            if len(window.scene_ids) > 1:
                current_transition = transition_representatives.get(window.timeline_region)
                transition_sort_key = (-len(window.scene_ids), *window.selection_score)
                if current_transition is None:
                    transition_representatives[window.timeline_region] = window
                else:
                    current_sort_key = (
                        -len(current_transition.scene_ids),
                        *current_transition.selection_score,
                    )
                    if transition_sort_key < current_sort_key:
                        transition_representatives[window.timeline_region] = window

    selected: list[CandidateWindow] = []
    for key in sorted(cell_representatives):
        window = cell_representatives[key]
        window.selection_reasons.append("duration_region_representative")
        selected.append(window)

    selected_keys = {_window_key(window) for window in selected}
    transition_count = 0
    for region in sorted(transition_representatives):
        window = transition_representatives[region]
        if _window_key(window) in selected_keys:
            continue
        window.selection_reasons.append("scene_transition_representative")
        selected.append(window)
        selected_keys.add(_window_key(window))
        transition_count += 1

    selected.sort(
        key=lambda window: (
            window.timeline_region,
            window.start_hundredths,
            window.end_hundredths,
            window.duration_band,
        )
    )
    deduplicated: list[CandidateWindow] = []
    for candidate in selected:
        duplicate = next(
            (
                existing
                for existing in deduplicated
                if _is_near_duplicate(candidate, existing)
            ),
            None,
        )
        if duplicate is None:
            deduplicated.append(candidate)
        else:
            duplicate.selection_reasons.append(
                "also_selected_as_scene_transition_representative"
                if "scene_transition_representative" in candidate.selection_reasons
                else "duplicate_selection_reason"
            )

    return (
        stage_a_count,
        stage_a_duration_distribution,
        deduplicated,
        transition_count,
        len(selected) - len(deduplicated),
    )


def _boundary_record(
    hundredths: int,
    boundary_references: dict[int, list[BoundaryReference]],
) -> dict[str, Any]:
    references = boundary_references[hundredths]
    raw = references[0].raw
    return {
        "raw": raw,
        "seconds": hundredths / 100,
        "source_event_timestamps": [
            {
                "event_id": reference.event_id,
                "scene_id": reference.scene_id,
                "field": reference.field,
                "raw": reference.raw,
            }
            for reference in references
        ],
    }


def _scene_summary(scene: dict[str, Any], scene_id: str) -> dict[str, Any]:
    heading = scene.get("scene_heading")
    return {
        "scene_id": scene_id,
        "scene_number": scene.get("number"),
        "raw_number": scene.get("raw_number"),
        "heading": heading.get("value") if isinstance(heading, dict) else None,
        "start": copy.deepcopy(scene.get("start")),
        "end": copy.deepcopy(scene.get("end")),
    }


def _event_type_counts(event_records: list[EventRecord], indices: tuple[int, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indices:
        event_type = str(event_records[index].event.get("type", "UNKNOWN"))
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _materialize_candidate(
    window: CandidateWindow,
    *,
    candidate_index: int,
    episode: dict[str, Any],
    episode_number: int | None,
    event_records: list[EventRecord],
    scene_ranges: list[tuple[str, int, int, dict[str, Any]]],
    boundary_references: dict[int, list[BoundaryReference]],
) -> dict[str, Any]:
    duration_hundredths = window.end_hundredths - window.start_hundredths
    duration_seconds = duration_hundredths / 100
    source_events = [
        {
            "scene_id": event_records[index].scene_id,
            "event": copy.deepcopy(event_records[index].event),
        }
        for index in window.event_indices
    ]
    intersecting_scene_data = [
        _scene_summary(scene, scene_id)
        for scene_id, scene_start, scene_end, scene in scene_ranges
        if scene_id in window.scene_ids
        and scene_start < window.end_hundredths
        and scene_end > window.start_hundredths
    ]
    for source_event in source_events:
        owner_id = source_event["scene_id"]
        if owner_id not in {scene["scene_id"] for scene in intersecting_scene_data}:
            owner = next(
                (
                    scene
                    for scene_id, _, _, scene in scene_ranges
                    if scene_id == owner_id
                ),
                None,
            )
            if owner is not None:
                intersecting_scene_data.append(_scene_summary(owner, owner_id))

    source_events.sort(
        key=lambda item: (
            item["event"].get("source_lines", {}).get("header", 0),
            item["event"].get("id", ""),
        )
    )
    event_ids = [item["event"].get("id") for item in source_events]
    counts = _event_type_counts(event_records, window.event_indices)
    region_start = window.timeline_region / TIMELINE_REGION_COUNT
    region_end = (window.timeline_region + 1) / TIMELINE_REGION_COUNT
    return {
        "candidate_id": f"{episode['episode_id']}:clip-{candidate_index:04d}",
        "episode": {
            "episode_id": episode["episode_id"],
            "episode_number": episode_number,
        },
        "time_range": {
            "start": _boundary_record(window.start_hundredths, boundary_references),
            "end": _boundary_record(window.end_hundredths, boundary_references),
            "duration_seconds": duration_seconds,
        },
        "timeline_region": {
            "index": window.timeline_region + 1,
            "label": f"region-{window.timeline_region + 1}",
            "fraction_start": region_start,
            "fraction_end": region_end,
        },
        "duration_band": window.duration_band,
        "scenes": intersecting_scene_data,
        "scene_count": len(intersecting_scene_data),
        "events": source_events,
        "event_ids": event_ids,
        "event_count": len(source_events),
        "event_type_counts": counts,
        "structural_signals": {
            "event_density_per_second": len(source_events) / duration_seconds,
            "character_dialogue_count": counts.get("CHARACTER_DIALOGUE", 0),
            "action_count": counts.get("ACTION", 0),
            "dialogue_action_balance": {
                "dialogue": counts.get("CHARACTER_DIALOGUE", 0),
                "action": counts.get("ACTION", 0),
            },
            "spans_multiple_scenes": len(intersecting_scene_data) > 1,
        },
        "selection_metadata": {
            "reasons": list(window.selection_reasons),
            "temporal_iou_threshold": TEMPORAL_IOU_THRESHOLD,
            "event_jaccard_threshold": EVENT_JACCARD_THRESHOLD,
        },
    }


def validate_candidate(
    candidate: dict[str, Any],
    *,
    source_episode_id: str,
    source_events_by_id: dict[str, tuple[str, dict[str, Any], int, int]],
    source_boundaries: set[int],
    all_source_events: list[EventRecord],
) -> list[str]:
    """Validate candidate range, episode, overlap, and source-event preservation."""
    errors: list[str] = []
    episode = candidate.get("episode")
    if not isinstance(episode, dict) or episode.get("episode_id") != source_episode_id:
        errors.append("candidate_does_not_belong_to_exactly_one_source_episode")
        return errors

    time_range = candidate.get("time_range", {})
    start = _timestamp_to_hundredths(
        {
            "raw": time_range.get("start", {}).get("raw"),
        }
    )
    end = _timestamp_to_hundredths(
        {
            "raw": time_range.get("end", {}).get("raw"),
        }
    )
    if start is None or end is None:
        errors.append("candidate_boundary_timestamp_invalid")
        return errors
    if start not in source_boundaries or end not in source_boundaries:
        errors.append("candidate_boundary_not_in_source_event_timestamps")
    duration = end - start
    if not MIN_DURATION_HUNDREDTHS <= duration <= MAX_DURATION_HUNDREDTHS:
        errors.append("candidate_duration_out_of_range")
    if abs(candidate.get("time_range", {}).get("duration_seconds", -1) - duration / 100) > 1e-9:
        errors.append("candidate_duration_value_mismatch")

    if not isinstance(candidate.get("events"), list) or not candidate["events"]:
        errors.append("candidate_has_no_events")
        return errors
    expected_event_ids = {
        str(record.event.get("id", ""))
        for record in all_source_events
        if record.start_hundredths < end and record.end_hundredths > start
    }
    actual_event_ids = {
        str(item.get("event", {}).get("id", "")) for item in candidate["events"]
    }
    if actual_event_ids != expected_event_ids:
        errors.append("candidate_event_set_does_not_match_strict_overlap")
    for item in candidate["events"]:
        event_id = item.get("event", {}).get("id")
        source_record = source_events_by_id.get(event_id)
        if source_record is None:
            errors.append(f"event_not_found_in_source:{event_id}")
            continue
        source_scene_id, source_event, event_start, event_end = source_record
        if item.get("scene_id") != source_scene_id:
            errors.append(f"event_scene_id_changed:{event_id}")
        if item.get("event") != source_event:
            errors.append(f"source_event_data_changed:{event_id}")
        if not (event_start < end and event_end > start):
            errors.append(f"event_does_not_strictly_overlap_candidate:{event_id}")
    return errors


def generate_episode_candidates(episode: dict[str, Any]) -> dict[str, Any]:
    """Generate Stage A and Stage B candidates for one parsed episode."""
    (
        event_records,
        scene_ranges,
        boundary_references,
        episode_duration,
        invalid_event_timestamp_count,
    ) = _collect_source_data(episode)
    boundary_values = sorted(boundary_references)
    (
        stage_a_count,
        stage_a_duration_distribution,
        selected_windows,
        transition_count,
        deduplicated_count,
    ) = _select_stage_b(event_records, scene_ranges, boundary_values, episode_duration)
    episode_number = _episode_number(episode)
    candidates = [
        _materialize_candidate(
            window,
            candidate_index=index,
            episode=episode,
            episode_number=episode_number,
            event_records=event_records,
            scene_ranges=scene_ranges,
            boundary_references=boundary_references,
        )
        for index, window in enumerate(selected_windows, start=1)
    ]

    source_events_by_id: dict[str, tuple[str, dict[str, Any], int, int]] = {}
    for record in event_records:
        event_id = str(record.event.get("id", ""))
        source_events_by_id[event_id] = (
            record.scene_id,
            record.event,
            record.start_hundredths,
            record.end_hundredths,
        )
    source_boundaries = set(boundary_references)
    validation_errors: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if candidate["candidate_id"] in candidate_ids:
            validation_errors.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "error": "duplicate_candidate_id",
                }
            )
        candidate_ids.add(candidate["candidate_id"])
        for error in validate_candidate(
            candidate,
            source_episode_id=str(episode["episode_id"]),
            source_events_by_id=source_events_by_id,
            source_boundaries=source_boundaries,
            all_source_events=event_records,
        ):
            validation_errors.append(
                {"candidate_id": candidate["candidate_id"], "error": error}
            )

    duration_distribution: dict[str, int] = {
        label: 0 for label, _, _, _ in DURATION_BANDS
    }
    scene_transition_count = 0
    for candidate in candidates:
        duration_distribution[candidate["duration_band"]] += 1
        if candidate["structural_signals"]["spans_multiple_scenes"]:
            scene_transition_count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode["episode_id"],
        "episode_number": episode_number,
        "generation_settings": {
            "minimum_duration_seconds": MIN_DURATION_HUNDREDTHS / 100,
            "maximum_duration_seconds": MAX_DURATION_HUNDREDTHS / 100,
            "timeline_regions": TIMELINE_REGION_COUNT,
            "duration_bands": [
                {"label": label, "minimum_hundredths": minimum, "maximum_hundredths": maximum}
                for label, minimum, maximum, _ in DURATION_BANDS
            ],
            "overlap_condition": "event_start < candidate_end and event_end > candidate_start",
            "temporal_iou_threshold": TEMPORAL_IOU_THRESHOLD,
            "event_jaccard_threshold": EVENT_JACCARD_THRESHOLD,
        },
        "counts": {
            "stage_a_candidates": stage_a_count,
            "stage_b_candidates_before_deduplication": len(selected_windows)
            + deduplicated_count,
            "stage_b_candidates": len(candidates),
            "scene_transition_candidates": scene_transition_count,
            "deduplicated_candidates": deduplicated_count,
            "scene_transition_representatives_added": transition_count,
            "multi_scene_candidates": scene_transition_count,
            "invalid_event_timestamps_skipped": invalid_event_timestamp_count,
            "stage_a_duration_distribution": stage_a_duration_distribution,
            "duration_distribution": duration_distribution,
        },
        "validation": {
            "passed": not validation_errors,
            "candidate_count": len(candidates),
            "errors": validation_errors,
        },
        "candidates": candidates,
    }


def _candidate_digest(candidate: dict[str, Any]) -> str:
    """Hash stable candidate content for summary-level reproducibility."""
    encoded = json.dumps(
        candidate,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_candidates(
    input_directory: str | Path,
    output_directory: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate per-episode candidate JSON and a batch summary JSON."""
    input_path = Path(input_directory)
    output_path = Path(output_directory)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Parsed JSON directory does not exist: {input_path}")

    json_files = sorted(
        (path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
        key=lambda path: [
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    output_path.mkdir(parents=True, exist_ok=True)

    episode_outputs: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    all_candidate_ids: set[str] = set()
    batch_validation_errors: list[dict[str, Any]] = []
    totals = {
        "stage_a_candidates": 0,
        "stage_b_candidates_before_deduplication": 0,
        "stage_b_candidates": 0,
        "scene_transition_candidates": 0,
        "scene_transition_representatives_added": 0,
        "multi_scene_candidates": 0,
        "deduplicated_candidates": 0,
        "invalid_event_timestamps_skipped": 0,
        "stage_a_duration_distribution": {
            label: 0 for label, _, _, _ in DURATION_BANDS
        },
        "duration_distribution": {label: 0 for label, _, _, _ in DURATION_BANDS},
    }
    per_episode: list[dict[str, Any]] = []
    for json_path in json_files:
        episode = _load_episode(json_path)
        episode_output = generate_episode_candidates(episode)
        episode_id = episode_output["episode_id"]
        if episode_id in seen_episode_ids:
            batch_validation_errors.append(
                {"episode_id": episode_id, "error": "duplicate_episode_id"}
            )
        seen_episode_ids.add(episode_id)

        episode_output_path = output_path / f"{json_path.stem}.json"
        episode_output_path.write_text(
            json.dumps(episode_output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        episode_outputs.append(episode_output)

        for candidate in episode_output["candidates"]:
            candidate_id = candidate["candidate_id"]
            if candidate_id in all_candidate_ids:
                batch_validation_errors.append(
                    {
                        "episode_id": episode_id,
                        "candidate_id": candidate_id,
                        "error": "duplicate_candidate_id_across_batch",
                    }
                )
            all_candidate_ids.add(candidate_id)

        for key in (
            "stage_a_candidates",
            "stage_b_candidates_before_deduplication",
            "stage_b_candidates",
            "scene_transition_candidates",
            "scene_transition_representatives_added",
            "multi_scene_candidates",
            "deduplicated_candidates",
            "invalid_event_timestamps_skipped",
        ):
            totals[key] += episode_output["counts"][key]
        for band, count in episode_output["counts"]["duration_distribution"].items():
            totals["duration_distribution"][band] += count
        for band, count in episode_output["counts"]["stage_a_duration_distribution"].items():
            totals["stage_a_duration_distribution"][band] += count
        if not episode_output["validation"]["passed"]:
            batch_validation_errors.extend(
                {
                    "episode_id": episode_id,
                    **error,
                }
                for error in episode_output["validation"]["errors"]
            )
        per_episode.append(
            {
                "episode_id": episode_id,
                "episode_number": episode_output["episode_number"],
                **episode_output["counts"],
                "validation_passed": episode_output["validation"]["passed"],
                "candidate_digest": _candidate_digest(
                    {
                        "episode_id": episode_id,
                        "candidates": episode_output["candidates"],
                    }
                ),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generation_settings": {
            "input_directory": str(input_path),
            "output_directory": str(output_path),
            "minimum_duration_seconds": MIN_DURATION_HUNDREDTHS / 100,
            "maximum_duration_seconds": MAX_DURATION_HUNDREDTHS / 100,
            "timeline_regions": TIMELINE_REGION_COUNT,
            "timeline_region_basis": "candidate_midpoint",
            "duration_bands": [
                {
                    "label": label,
                    "minimum_seconds": minimum / 100,
                    "maximum_seconds": maximum / 100,
                    "maximum_inclusive": maximum == MAX_DURATION_HUNDREDTHS,
                }
                for label, minimum, maximum, _ in DURATION_BANDS
            ],
            "boundary_source": "distinct parsed event start/end timestamps",
            "overlap_condition": "event_start < candidate_end and event_end > candidate_start",
            "transition_representatives": "up to one additional multi-scene candidate per timeline region",
            "near_duplicate_rule": {
                "same_episode": True,
                "same_timeline_region": True,
                "same_duration_band": True,
                "same_scene_set": True,
                "temporal_iou_minimum": TEMPORAL_IOU_THRESHOLD,
                "event_id_jaccard_minimum": EVENT_JACCARD_THRESHOLD,
                "tie_break": "timeline region, start hundredths, end hundredths, duration band",
            },
        },
        "counts": {
            "parsed_episode_files": len(json_files),
            "unique_episode_ids": len(seen_episode_ids),
            **totals,
        },
        "per_episode": per_episode,
        "validation": {
            "passed": not batch_validation_errors
            and all(item["validation_passed"] for item in per_episode),
            "candidate_ids_unique": len(all_candidate_ids)
            == sum(len(item["candidates"]) for item in episode_outputs),
            "errors": batch_validation_errors,
        },
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return episode_outputs, summary


def _default_input_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "output" / "parsed"


def _default_output_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "output" / "candidates"


def main(argv: list[str] | None = None) -> int:
    """Run candidate generation from the project root or any working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_directory(),
        help="Directory containing parsed episode JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_directory(),
        help="Directory for per-episode candidates and summary.json.",
    )
    arguments = parser.parse_args(argv)
    outputs, summary = generate_candidates(arguments.input_dir, arguments.output_dir)
    print(
        f"Generated candidates for {len(outputs)} episode(s): "
        f"Stage A={summary['counts']['stage_a_candidates']}, "
        f"Stage B={summary['counts']['stage_b_candidates']}, "
        f"validation={'PASS' if summary['validation']['passed'] else 'FAIL'}."
    )
    return 0 if summary["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
