"""Parse structured screenplay TXT files into source-preserving JSON."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SCHEMA_VERSION = "1.0"
TIMESTAMP_PATTERN = re.compile(r"^(?P<minutes>\d{2}):(?P<seconds>\d{2})\.(?P<hundredths>\d{2})$")
METADATA_KEYS = {
    "EPISODE NUMBER": "episode_number",
    "EPISODE LOGLINE": "episode_logline",
    "CONTINUITY NOTES": "continuity_notes",
}
KNOWN_EVENT_TYPES = {
    "CAMERA": "CAMERA",
    "ACTION": "ACTION",
    "SOUND FX": "SOUND_FX",
}
SCENE_MARKER_PATTERN = re.compile(r"^\s*SCENE\s+(?P<number>\d+)\s*$", re.IGNORECASE)
SCENE_FIELD_PATTERN = re.compile(
    r"^\s*(?P<field>SCENE\s+HEADING|SCENE\s+START|SCENE\s+END|"
    r"ACTIONS\s+AND\s+DIALOGUE)\s*:\s*(?P<value>.*)$",
    re.IGNORECASE,
)
EVENT_HEADER_PATTERN = re.compile(r"^\s*\*\*(?P<label>.*?)\*\*\s*$")
COMBINED_TIMESTAMP_PATTERN = re.compile(
    r"^\s*Start\s*:\s*(?P<start>.*?)\s*\|\s*End\s*:\s*(?P<end>.*?)\s*$",
    re.IGNORECASE,
)
SINGLE_TIMESTAMP_PATTERN = re.compile(
    r"^\s*(?P<field>Start|End)\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
METADATA_PATTERN = re.compile(
    r"^\ufeff?\s*(?P<key>EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)"
    r"\s*:\s*(?P<value>.*)$",
    re.IGNORECASE,
)
SCRIPT_MARKER_PATTERN = re.compile(
    r"^\s*---\s*SCRIPT\s+(?P<kind>START|END)\s*---\s*$",
    re.IGNORECASE,
)
FILENAME_EPISODE_PATTERN = re.compile(r"(?:^|[^a-z])episode[-_ ]?(?P<number>\d+)", re.IGNORECASE)
RESERVED_SPEAKER_LABELS = {
    "DIALOGUE",
    "EVENT",
    "SCENE",
    "SCENE END",
    "SCENE HEADING",
    "SCENE START",
    "START",
    "END",
    "VOICE OVER",
}


@dataclass(frozen=True)
class SourceLine:
    """A source line, retaining its original line ending."""

    number: int
    text: str
    ending: str
    raw: str


@dataclass(frozen=True)
class EventClassification:
    """Classification and label components for a bold event header."""

    event_type: Literal["CAMERA", "ACTION", "SOUND_FX", "CHARACTER_DIALOGUE", "UNKNOWN"]
    speaker_label_raw: str | None
    annotations_raw: list[str]
    is_voice_over: bool


def _split_source_lines(text: str) -> list[SourceLine]:
    """Split decoded text into lines without losing original line endings."""
    result: list[SourceLine] = []
    for number, raw in enumerate(text.splitlines(keepends=True), start=1):
        if raw.endswith("\r\n"):
            line_text, ending = raw[:-2], "\r\n"
        elif raw.endswith(("\n", "\r")):
            line_text, ending = raw[:-1], raw[-1]
        else:
            line_text, ending = raw, ""
        result.append(SourceLine(number=number, text=line_text, ending=ending, raw=raw))
    return result


def _timestamp_raw(value: str | None) -> str | None:
    """Remove only a balanced pair of wrapping parentheses from a timestamp."""
    if value is None:
        return None
    raw = value.strip()
    if raw.startswith("(") and raw.endswith(")"):
        return raw[1:-1].strip()
    return raw


def parse_timestamp(raw_timestamp: str | None) -> dict[str, Any]:
    """Parse MM:SS.hh into seconds while preserving its raw value.

    The returned ``raw`` field excludes surrounding parentheses when present.
    Invalid or missing values have ``seconds`` set to ``None``.
    """
    raw = _timestamp_raw(raw_timestamp)
    if raw is None:
        return {"raw": None, "seconds": None, "valid": False}

    match = TIMESTAMP_PATTERN.fullmatch(raw)
    if match is None:
        return {"raw": raw, "seconds": None, "valid": False}

    seconds_component = int(match.group("seconds"))
    if seconds_component >= 60:
        return {"raw": raw, "seconds": None, "valid": False}

    total_seconds = (
        int(match.group("minutes")) * 60
        + seconds_component
        + int(match.group("hundredths")) / 100
    )
    return {"raw": raw, "seconds": total_seconds, "valid": True}


def validate_timestamp_pair(
    start: dict[str, Any],
    end: dict[str, Any],
    *,
    scope: str,
    source_line: int | None = None,
) -> list[dict[str, Any]]:
    """Return timestamp validation findings for a start/end pair."""
    findings: list[dict[str, Any]] = []
    for name, value in (("start", start), ("end", end)):
        if value.get("raw") is None:
            findings.append(
                _finding(
                    "missing_timestamp",
                    "error",
                    f"{scope} is missing its {name} timestamp.",
                    source_line=source_line,
                    timestamp_field=name,
                )
            )
        elif not value.get("valid", value.get("seconds") is not None):
            findings.append(
                _finding(
                    "invalid_timestamp",
                    "error",
                    f"{scope} has an invalid {name} timestamp.",
                    source_line=value.get("source_line", source_line),
                    timestamp_field=name,
                    raw_timestamp=value.get("raw"),
                )
            )

    start_seconds, end_seconds = start.get("seconds"), end.get("seconds")
    if start_seconds is not None and end_seconds is not None and start_seconds > end_seconds:
        findings.append(
            _finding(
                "start_after_end",
                "error",
                f"{scope} starts after it ends.",
                source_line=source_line,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return findings


def _extract_parenthetical_annotations(label: str) -> tuple[str, list[str], bool]:
    """Extract balanced parenthetical groups without interpreting their meaning."""
    base_parts: list[str] = []
    annotations: list[str] = []
    depth = 0
    annotation_start: int | None = None
    cursor = 0

    while cursor < len(label):
        character = label[cursor]
        if character == "(":
            if depth == 0:
                annotation_start = cursor
                base_parts.append(" ")
            depth += 1
        elif character == ")" and depth:
            depth -= 1
            if depth == 0 and annotation_start is not None:
                annotations.append(label[annotation_start : cursor + 1])
                annotation_start = None
        elif depth == 0:
            base_parts.append(character)
        cursor += 1

    balanced = depth == 0
    if not balanced:
        return label, annotations, False

    base = re.sub(r"\s+", " ", "".join(base_parts)).strip()
    return base, annotations, True


def classify_event_label(raw_label: str) -> EventClassification:
    """Classify a bold header as a known event, dialogue, or UNKNOWN.

    Dialogue is recognized only when the non-parenthetical label follows the
    dataset's uppercase speaker-label convention and is not a reserved marker.
    The original label and every parenthetical annotation remain available.
    """
    label_for_match = raw_label.strip()
    normalized = re.sub(r"\s+", " ", label_for_match).upper()
    if normalized in KNOWN_EVENT_TYPES:
        return EventClassification(
            event_type=KNOWN_EVENT_TYPES[normalized],  # type: ignore[arg-type]
            speaker_label_raw=None,
            annotations_raw=[],
            is_voice_over=False,
        )

    base, annotations, balanced = _extract_parenthetical_annotations(raw_label)
    speaker_candidate = base.strip()
    valid_speaker = (
        balanced
        and bool(speaker_candidate)
        and speaker_candidate.upper() == speaker_candidate
        and speaker_candidate not in RESERVED_SPEAKER_LABELS
        and re.fullmatch(r"[A-Z0-9][A-Z0-9 .&'’/-]*", speaker_candidate) is not None
        and any(character.isalpha() for character in speaker_candidate)
    )
    if valid_speaker:
        is_voice_over = any(
            re.search(r"\bV\.O\.", annotation, re.IGNORECASE) is not None
            for annotation in annotations
        )
        return EventClassification(
            event_type="CHARACTER_DIALOGUE",
            speaker_label_raw=raw_label,
            annotations_raw=annotations,
            is_voice_over=is_voice_over,
        )

    return EventClassification(
        event_type="UNKNOWN",
        speaker_label_raw=None,
        annotations_raw=annotations,
        is_voice_over=any(
            re.search(r"\bV\.O\.", annotation, re.IGNORECASE) is not None
            for annotation in annotations
        ),
    )


def _finding(
    code: str,
    severity: str,
    message: str,
    *,
    source_line: int | None = None,
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if source_line is not None:
        result["source_line"] = source_line
    if details:
        result["details"] = details
    return result


def _empty_timestamp() -> dict[str, Any]:
    return {
        "raw": None,
        "seconds": None,
        "original_line": None,
        "original_line_ending": None,
        "source_line": None,
        "valid": False,
    }


def _make_timestamp(value: str, line: SourceLine) -> dict[str, Any]:
    parsed = parse_timestamp(value)
    return {
        **parsed,
        "original_line": line.text,
        "original_line_ending": line.ending,
        "source_line": line.number,
    }


def _add_scene_field(
    scene: dict[str, Any],
    field_name: str,
    value: str,
    line: SourceLine,
    findings: list[dict[str, Any]],
) -> None:
    field_record: dict[str, Any] = {
        "field": field_name,
        "value": value,
        "raw_line": line.text,
        "line_ending": line.ending,
        "source_line": line.number,
    }
    scene["fields"].append(field_record)
    if field_name in {"scene_heading", "scene_start", "scene_end"}:
        existing = scene[field_name]
        if existing is not None:
            findings.append(
                _finding(
                    "duplicate_scene_field",
                    "warning",
                    f"Scene contains more than one {field_name.replace('_', ' ')} field.",
                    source_line=line.number,
                    scene_id=scene["id"],
                )
            )
        else:
            scene[field_name] = field_record


def _timestamp_line_values(line: SourceLine) -> tuple[dict[str, str], list[str]] | None:
    """Return field values and field names when a line is timestamp structure."""
    combined = COMBINED_TIMESTAMP_PATTERN.fullmatch(line.text)
    if combined is not None:
        return (
            {"start": combined.group("start"), "end": combined.group("end")},
            ["start", "end"],
        )
    single = SINGLE_TIMESTAMP_PATTERN.fullmatch(line.text)
    if single is not None:
        field = single.group("field").lower()
        return ({field: single.group("value")}, [field])
    return None


def _parse_event_block(
    lines: list[SourceLine],
    header_index: int,
    block_end: int,
    *,
    episode_id: str,
    scene_index: int,
    event_index: int,
    recognized_lines: set[int],
    content_lines: set[int],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    header = lines[header_index]
    match = EVENT_HEADER_PATTERN.fullmatch(header.text)
    assert match is not None
    raw_label = match.group("label")
    if raw_label.endswith(":"):
        raw_label = raw_label[:-1]
    classification = classify_event_label(raw_label)
    event_id = f"{episode_id}:scene-{scene_index:03d}:event-{event_index:03d}"
    body = lines[header_index + 1 : block_end]

    timestamps: dict[str, dict[str, Any]] = {
        "start": _empty_timestamp(),
        "end": _empty_timestamp(),
    }
    timestamp_lines: list[dict[str, Any]] = []
    timestamp_line_numbers: set[int] = set()
    for line in body:
        parsed_line = _timestamp_line_values(line)
        if parsed_line is None:
            continue
        values, fields = parsed_line
        recognized_lines.add(line.number)
        timestamp_line_numbers.add(line.number)
        timestamp_lines.append(
            {
                "raw_line": line.text,
                "line_ending": line.ending,
                "source_line": line.number,
                "fields": fields,
            }
        )
        for field, raw_value in values.items():
            if timestamps[field]["source_line"] is not None:
                findings.append(
                    _finding(
                        "duplicate_event_timestamp",
                        "warning",
                        f"Event contains more than one {field} timestamp.",
                        source_line=line.number,
                        event_id=event_id,
                    )
                )
                continue
            timestamps[field] = _make_timestamp(raw_value, line)

    content_positions = [
        position
        for position, line in enumerate(body)
        if line.number not in timestamp_line_numbers and line.text.strip()
    ]
    if content_positions:
        first_content, last_content = content_positions[0], content_positions[-1]
        selected_lines = [
            line
            for line in body[first_content : last_content + 1]
            if line.number not in timestamp_line_numbers
        ]
        raw_content = "".join(line.raw for line in selected_lines)
        content_source_lines = [line.number for line in selected_lines]
        content_lines.update(content_source_lines)
    else:
        raw_content = ""
        content_source_lines = []

    recognized_lines.add(header.number)
    event: dict[str, Any] = {
        "id": event_id,
        "index": event_index,
        "type": classification.event_type,
        "classification": {
            "rule": (
                "exact_known_event_label"
                if classification.event_type in {"CAMERA", "ACTION", "SOUND_FX"}
                else "uppercase_speaker_label"
                if classification.event_type == "CHARACTER_DIALOGUE"
                else "unrecognized_bold_label"
            ),
            "confidence": "high" if classification.event_type != "UNKNOWN" else "unresolved",
        },
        "raw_header_line": header.text,
        "raw_header_line_ending": header.ending,
        "raw_label": raw_label,
        "speaker_label_raw": classification.speaker_label_raw,
        "annotations_raw": classification.annotations_raw,
        "is_voice_over": classification.is_voice_over,
        "timestamp_lines": timestamp_lines,
        "start": timestamps["start"],
        "end": timestamps["end"],
        "content": {
            "kind": (
                "dialogue"
                if classification.event_type == "CHARACTER_DIALOGUE"
                else "description"
                if classification.event_type in {"CAMERA", "ACTION", "SOUND_FX"}
                else "unknown"
            ),
            "raw": raw_content,
            "source_lines": content_source_lines,
        },
        "source_lines": {
            "header": header.number,
            "body_start": body[0].number if body else None,
            "body_end": body[-1].number if body else None,
        },
        "validation_flags": [],
    }

    findings.extend(
        validate_timestamp_pair(
            event["start"],
            event["end"],
            scope=f"Event {event_id}",
            source_line=header.number,
        )
    )
    if not raw_content:
        findings.append(
            _finding(
                "missing_event_content",
                "error",
                f"Event {event_id} has no description or dialogue content.",
                source_line=header.number,
                event_id=event_id,
            )
        )
    if classification.event_type == "UNKNOWN":
        findings.append(
            _finding(
                "unknown_event_label",
                "warning",
                f"Could not confidently classify bold label {raw_label!r}.",
                source_line=header.number,
                event_id=event_id,
                raw_label=raw_label,
            )
        )
    return event


def _parse_scene_segment(
    lines: list[SourceLine],
    scene_start: int,
    scene_end: int,
    *,
    episode_id: str,
    scene_index: int,
    scene_number_raw: str,
    recognized_lines: set[int],
    content_lines: set[int],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    scene_id = f"{episode_id}:scene-{scene_index:03d}"
    scene: dict[str, Any] = {
        "id": scene_id,
        "index": scene_index,
        "number": int(scene_number_raw),
        "raw_number": scene_number_raw,
        "source_lines": {
            "start": lines[scene_start].number,
            "end": lines[scene_end - 1].number if scene_end > scene_start else lines[scene_start].number,
        },
        "fields": [],
        "scene_heading": None,
        "scene_start": None,
        "scene_end": None,
        "section_markers": [],
        "events": [],
    }
    marker = lines[scene_start]
    recognized_lines.add(marker.number)

    event_header_indices: list[int] = []
    scene_field_lines: dict[int, tuple[str, str]] = {}
    for position in range(scene_start + 1, scene_end):
        line = lines[position]
        field_match = SCENE_FIELD_PATTERN.fullmatch(line.text)
        if field_match is not None:
            label = re.sub(r"\s+", " ", field_match.group("field").upper())
            value = field_match.group("value")
            field_name = {
                "SCENE HEADING": "scene_heading",
                "SCENE START": "scene_start",
                "SCENE END": "scene_end",
                "ACTIONS AND DIALOGUE": "actions_and_dialogue",
            }[label]
            recognized_lines.add(line.number)
            scene_field_lines[position] = (field_name, value)
            if field_name == "actions_and_dialogue":
                scene["section_markers"].append(
                    {
                        "raw_line": line.text,
                        "line_ending": line.ending,
                        "source_line": line.number,
                    }
                )
            else:
                _add_scene_field(scene, field_name, value, line, findings)
            continue

        if EVENT_HEADER_PATTERN.fullmatch(line.text) is not None:
            event_header_indices.append(position)

    for field_key in ("scene_heading", "scene_start", "scene_end"):
        if scene[field_key] is None:
            findings.append(
                _finding(
                    f"missing_{field_key}",
                    "error",
                    f"Scene {scene_id} is missing its {field_key.replace('_', ' ')} field.",
                    source_line=marker.number,
                    scene_id=scene_id,
                )
            )

    scene_start_timestamp = _empty_timestamp()
    scene_end_timestamp = _empty_timestamp()
    for field_key, target in (
        ("scene_start", scene_start_timestamp),
        ("scene_end", scene_end_timestamp),
    ):
        field_record = scene[field_key]
        if field_record is not None:
            parsed = parse_timestamp(field_record["value"])
            target.update(
                {
                    **parsed,
                    "original_line": field_record["raw_line"],
                    "original_line_ending": field_record["line_ending"],
                    "source_line": field_record["source_line"],
                }
            )
    scene["start"] = scene_start_timestamp
    scene["end"] = scene_end_timestamp
    findings.extend(
        validate_timestamp_pair(
            scene["start"],
            scene["end"],
            scope=f"Scene {scene_id}",
            source_line=marker.number,
        )
    )

    for index, header_position in enumerate(event_header_indices, start=1):
        next_event_position = (
            event_header_indices[index] if index < len(event_header_indices) else scene_end
        )
        structural_positions = [
            position for position in scene_field_lines if header_position < position < next_event_position
        ]
        block_end = min(structural_positions) if structural_positions else next_event_position
        event = _parse_event_block(
            lines,
            header_position,
            block_end,
            episode_id=episode_id,
            scene_index=scene_index,
            event_index=index,
            recognized_lines=recognized_lines,
            content_lines=content_lines,
            findings=findings,
        )
        scene["events"].append(event)

        start_seconds, end_seconds = event["start"]["seconds"], event["end"]["seconds"]
        scene_start_seconds, scene_end_seconds = scene["start"]["seconds"], scene["end"]["seconds"]
        if (
            start_seconds is not None
            and end_seconds is not None
            and scene_start_seconds is not None
            and scene_end_seconds is not None
            and (start_seconds < scene_start_seconds or end_seconds > scene_end_seconds)
        ):
            event["validation_flags"].append("event_crosses_scene_boundary")
            findings.append(
                _finding(
                    "event_crosses_scene_boundary",
                    "warning",
                    f"Event {event['id']} extends outside its scene time range.",
                    source_line=event["start"]["source_line"],
                    event_id=event["id"],
                    scene_id=scene_id,
                    event_start=start_seconds,
                    event_end=end_seconds,
                    scene_start=scene_start_seconds,
                    scene_end=scene_end_seconds,
                )
            )
    return scene


def _episode_id(filename: str) -> tuple[str, int | None]:
    match = FILENAME_EPISODE_PATTERN.search(filename)
    if match is None:
        return Path(filename).stem, None
    episode_number = int(match.group("number"))
    return f"episode-{episode_number:03d}", episode_number


def _metadata_number(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\bEpisode\s+(\d+)\b", value, re.IGNORECASE)
    if match is None:
        match = re.fullmatch(r"\s*(\d+)\s*", value)
    return int(match.group(1)) if match is not None else None


def parse_screenplay_bytes(
    source_bytes: bytes,
    *,
    filename: str,
) -> dict[str, Any]:
    """Parse one screenplay from bytes while retaining an exact source copy."""
    text = source_bytes.decode("utf-8")
    lines = _split_source_lines(text)
    episode_id, filename_episode_number = _episode_id(filename)
    recognized_lines: set[int] = set()
    content_lines: set[int] = set()
    findings: list[dict[str, Any]] = []
    metadata_occurrences: dict[str, list[dict[str, Any]]] = {
        value: [] for value in METADATA_KEYS.values()
    }
    script_markers: dict[str, list[dict[str, Any]]] = {"start": [], "end": []}

    for line in lines:
        metadata_match = METADATA_PATTERN.fullmatch(line.text)
        if metadata_match is not None:
            key = METADATA_KEYS[metadata_match.group("key").upper()]
            metadata_occurrences[key].append(
                {
                    "value": metadata_match.group("value"),
                    "raw_line": line.text,
                    "line_ending": line.ending,
                    "source_line": line.number,
                }
            )
            recognized_lines.add(line.number)
            continue

        marker_match = SCRIPT_MARKER_PATTERN.fullmatch(line.text)
        if marker_match is not None:
            kind = marker_match.group("kind").lower()
            script_markers[kind].append(
                {
                    "raw_line": line.text,
                    "line_ending": line.ending,
                    "source_line": line.number,
                }
            )
            recognized_lines.add(line.number)

    scene_markers: list[tuple[int, str]] = []
    for position, line in enumerate(lines):
        marker_match = SCENE_MARKER_PATTERN.fullmatch(line.text)
        if marker_match is not None:
            scene_markers.append((position, marker_match.group("number")))

    script_end_positions = [
        entry["source_line"] - 1 for entry in script_markers["end"]
    ]
    first_script_start = (
        script_markers["start"][0]["source_line"] - 1
        if script_markers["start"]
        else None
    )
    script_end_position = min(script_end_positions) if script_end_positions else len(lines)
    scenes: list[dict[str, Any]] = []
    for index, (scene_start, raw_number) in enumerate(scene_markers, start=1):
        if scene_start >= script_end_position:
            findings.append(
                _finding(
                    "scene_outside_script",
                    "warning",
                    "Scene marker appears after SCRIPT END.",
                    source_line=lines[scene_start].number,
                )
            )
            continue
        next_scene = (
            scene_markers[index][0]
            if index < len(scene_markers)
            else script_end_position
        )
        segment_end = min(next_scene, script_end_position)
        if segment_end <= scene_start:
            continue
        scene = _parse_scene_segment(
            lines,
            scene_start,
            segment_end,
            episode_id=episode_id,
            scene_index=len(scenes) + 1,
            scene_number_raw=raw_number,
            recognized_lines=recognized_lines,
            content_lines=content_lines,
            findings=findings,
        )
        if first_script_start is not None and scene_start < first_script_start:
            findings.append(
                _finding(
                    "scene_outside_script",
                    "warning",
                    "Scene marker appears before SCRIPT START.",
                    source_line=lines[scene_start].number,
                    scene_id=scene["id"],
                )
            )
        scenes.append(scene)

    for key, occurrences in metadata_occurrences.items():
        if not occurrences:
            findings.append(
                _finding(
                    "missing_metadata",
                    "error",
                    f"Required metadata field {key!r} is missing.",
                )
            )
        elif len(occurrences) > 1:
            findings.append(
                _finding(
                    "duplicate_metadata",
                    "warning",
                    f"Metadata field {key!r} appears more than once.",
                    source_line=occurrences[1]["source_line"],
                    field=key,
                    occurrence_count=len(occurrences),
                )
            )
    metadata = {
        key: (occurrences[0] if occurrences else None)
        for key, occurrences in metadata_occurrences.items()
    }

    for kind, occurrences in script_markers.items():
        if len(occurrences) != 1:
            findings.append(
                _finding(
                    f"script_marker_count_{kind}",
                    "error",
                    f"Expected exactly one SCRIPT {kind.upper()} marker; found {len(occurrences)}.",
                    occurrence_count=len(occurrences),
                )
            )
    if (
        script_markers["start"]
        and script_markers["end"]
        and script_markers["start"][0]["source_line"]
        >= script_markers["end"][0]["source_line"]
    ):
        findings.append(
            _finding(
                "script_markers_out_of_order",
                "error",
                "SCRIPT START does not precede SCRIPT END.",
                source_line=script_markers["start"][0]["source_line"],
            )
        )
    if not scenes:
        findings.append(_finding("no_scenes", "error", "No scene markers were found."))

    scene_numbers = [scene["number"] for scene in scenes]
    if len(set(scene_numbers)) != len(scene_numbers):
        findings.append(
            _finding(
                "duplicate_scene_number",
                "warning",
                "Scene numbering contains duplicates.",
                scene_numbers=scene_numbers,
            )
        )
    if any(right <= left for left, right in zip(scene_numbers, scene_numbers[1:])):
        findings.append(
            _finding(
                "scene_number_order",
                "warning",
                "Scene numbers are not strictly increasing.",
                scene_numbers=scene_numbers,
            )
        )

    for previous, current in zip(scenes, scenes[1:]):
        previous_end = previous["end"]["seconds"]
        current_start = current["start"]["seconds"]
        if previous_end is None or current_start is None:
            continue
        difference = round(current_start - previous_end, 2)
        if difference > 0:
            findings.append(
                _finding(
                    "scene_time_gap",
                    "warning",
                    "There is a gap between consecutive scene time ranges.",
                    scene_id=current["id"],
                    gap_seconds=difference,
                )
            )
        elif difference < 0:
            findings.append(
                _finding(
                    "scene_time_overlap",
                    "warning",
                    "Consecutive scene time ranges overlap.",
                    scene_id=current["id"],
                    overlap_seconds=abs(difference),
                )
            )

    internal_episode_number = _metadata_number(
        metadata["episode_number"]["value"] if metadata["episode_number"] else None
    )
    if metadata["episode_number"] and internal_episode_number is None:
        findings.append(
            _finding(
                "invalid_episode_number",
                "error",
                "EPISODE NUMBER does not contain a recognizable episode number.",
                source_line=metadata["episode_number"]["source_line"],
                value=metadata["episode_number"]["value"],
            )
        )
    if (
        filename_episode_number is not None
        and internal_episode_number is not None
        and filename_episode_number != internal_episode_number
    ):
        findings.append(
            _finding(
                "episode_number_mismatch",
                "error",
                "Filename episode number does not match the internal episode number.",
                filename_episode_number=filename_episode_number,
                internal_episode_number=internal_episode_number,
            )
        )

    formatting_separator_lines: list[int] = []
    unparsed_lines: list[dict[str, Any]] = []
    for line in lines:
        if line.number in recognized_lines or line.number in content_lines:
            continue
        if not line.text.strip():
            formatting_separator_lines.append(line.number)
            continue
        unparsed_lines.append(
            {
                "raw_line": line.text,
                "line_ending": line.ending,
                "source_line": line.number,
            }
        )
        findings.append(
            _finding(
                "unparsed_nonempty_line",
                "warning",
                "Non-empty source line was not recognized as metadata, structure, formatting, or event content.",
                source_line=line.number,
                raw_line=line.text,
            )
        )

    max_scene_end = max(
        (
            scene["end"]["seconds"]
            for scene in scenes
            if scene["end"]["seconds"] is not None
        ),
        default=None,
    )
    if max_scene_end is not None:
        for scene in scenes:
            for event in scene["events"]:
                event_end = event["end"]["seconds"]
                if event_end is not None and event_end > max_scene_end:
                    findings.append(
                        _finding(
                            "event_outside_episode_duration",
                            "warning",
                            f"Event {event['id']} ends after the final scene boundary.",
                            source_line=event["end"]["source_line"],
                            event_id=event["id"],
                            episode_end=max_scene_end,
                            event_end=event_end,
                        )
                    )

    retained_source_bytes = base64.b64decode(
        base64.b64encode(source_bytes), validate=True
    )
    source_bytes_match = retained_source_bytes == source_bytes
    computed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if not source_bytes_match:
        findings.append(
            _finding(
                "source_bytes_mismatch",
                "error",
                "Retained Base64 source bytes do not match the input bytes.",
            )
        )

    status = (
        "error"
        if any(finding["severity"] == "error" for finding in findings)
        else "warnings"
        if findings
        else "valid"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "source": {
            "filename": filename,
            "byte_length": len(source_bytes),
            "sha256": computed_sha256,
            "encoding": "utf-8",
            "original_bytes_base64": base64.b64encode(source_bytes).decode("ascii"),
            "source_bytes_match": source_bytes_match,
        },
        "metadata": metadata,
        "metadata_occurrences": metadata_occurrences,
        "script_markers": script_markers,
        "scenes": scenes,
        "formatting_separator_lines": formatting_separator_lines,
        "unparsed_lines": unparsed_lines,
        "validation": {
            "status": status,
            "findings": findings,
            "source_bytes_match": source_bytes_match,
            "source_byte_length_match": len(retained_source_bytes) == len(source_bytes),
            "source_sha256_match": hashlib.sha256(retained_source_bytes).hexdigest()
            == computed_sha256,
        },
    }


def parse_screenplay_file(file_path: str | Path) -> dict[str, Any]:
    """Read and parse one screenplay file."""
    path = Path(file_path)
    source_bytes = path.read_bytes()
    return parse_screenplay_bytes(source_bytes, filename=path.name)


def serialize_episode(episode: dict[str, Any]) -> str:
    """Serialize a parsed screenplay as readable UTF-8-compatible JSON."""
    return json.dumps(episode, ensure_ascii=False, indent=2) + "\n"


def write_episode_json(episode: dict[str, Any], output_directory: str | Path) -> Path:
    """Write one parsed episode to an output directory and return its path."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = str(episode["source"]["filename"])
    destination = output_path / f"{Path(filename).stem}.json"
    destination.write_text(serialize_episode(episode), encoding="utf-8", newline="\n")
    return destination


def parse_screenplay_directory(
    input_directory: str | Path,
    output_directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Parse all top-level TXT files in a directory and write their JSON output."""
    input_path = Path(input_directory)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
    output_path = (
        Path(output_directory)
        if output_directory is not None
        else Path(__file__).resolve().parents[2] / "data" / "output" / "parsed"
    )
    files = sorted(
        (path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: [
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    episodes: list[dict[str, Any]] = []
    for path in files:
        episode = parse_screenplay_file(path)
        write_episode_json(episode, output_path)
        episodes.append(episode)
    return episodes


def _default_output_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "output" / "parsed"


def main(argv: list[str] | None = None) -> int:
    """Run the parser for one file or all TXT files in a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file", type=Path, help="Parse one screenplay TXT file.")
    source_group.add_argument(
        "--input-dir",
        type=Path,
        help="Parse all top-level TXT files in a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_directory(),
        help="Directory for JSON output (default: data/output/parsed).",
    )
    arguments = parser.parse_args(argv)

    if arguments.file is not None:
        episode = parse_screenplay_file(arguments.file)
        output_file = write_episode_json(episode, arguments.output_dir)
        print(f"Wrote {output_file}")
        return 0

    episodes = parse_screenplay_directory(arguments.input_dir, arguments.output_dir)
    print(f"Parsed {len(episodes)} screenplay file(s) into {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
