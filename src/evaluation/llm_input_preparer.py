"""Prepare chronologically ordered, source-traceable candidate inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TIMESTAMP_PATTERN = re.compile(
    r"^(?P<minutes>\d{2}):(?P<seconds>\d{2})\.(?P<hundredths>\d{2})$"
)
SCHEMA_VERSION = "1.0"


def _timestamp_hundredths(timestamp: dict[str, Any], *, event_id: str, field: str) -> int:
    """Return exact hundredths for a valid MM:SS.hh event timestamp."""
    raw = timestamp.get("raw")
    match = TIMESTAMP_PATTERN.fullmatch(raw) if isinstance(raw, str) else None
    if match is None or int(match.group("seconds")) >= 60:
        raise ValueError(f"{event_id} has invalid {field} timestamp: {raw!r}")
    return (
        int(match.group("minutes")) * 6_000
        + int(match.group("seconds")) * 100
        + int(match.group("hundredths"))
    )


def _source_order(event: dict[str, Any], original_index: int) -> tuple[int, int]:
    """Use source header line as tie-breaker, falling back to input sequence."""
    lines = event.get("source_lines")
    header_line = lines.get("header") if isinstance(lines, dict) else None
    if isinstance(header_line, int):
        return (0, header_line)
    return (1, original_index)


def _event_sort_key(
    item: dict[str, Any], original_index: int
) -> tuple[int, int, int, int, int]:
    event = item.get("event")
    if not isinstance(event, dict):
        raise ValueError(f"Candidate event at index {original_index} is malformed")
    event_id = event.get("id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError(f"Candidate event at index {original_index} has no event ID")
    start = _timestamp_hundredths(event.get("start", {}), event_id=event_id, field="start")
    end = _timestamp_hundredths(event.get("end", {}), event_id=event_id, field="end")
    source_order = _source_order(event, original_index)
    return (start, end, source_order[0], source_order[1], original_index)


def prepare_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Copy one candidate and sort its unmodified event records chronologically."""
    prepared = copy.deepcopy(candidate)
    events = prepared.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{candidate.get('candidate_id', '<unknown>')} has no event list")
    indexed_events = list(enumerate(events))
    ordered = [
        item
        for _, item in sorted(
            indexed_events,
            key=lambda pair: _event_sort_key(pair[1], pair[0]),
        )
    ]
    changed = any(old is not new for old, new in zip(events, ordered, strict=True))
    prepared["events"] = ordered
    return prepared, changed


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject unexpected top-level values."""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(data: bytes) -> str:
    """Calculate a SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def _prepare_episode(
    candidate_path: Path,
    parsed_path: Path,
    screenplay_directory: Path,
    output_path: Path,
) -> tuple[int, int, int]:
    """Prepare a single episode file and write its ordered candidate records."""
    candidate_bytes = candidate_path.read_bytes()
    parsed_bytes = parsed_path.read_bytes()
    source = _read_json(candidate_path)
    parsed = _read_json(parsed_path)
    if source.get("episode_id") != parsed.get("episode_id"):
        raise ValueError(f"Episode ID mismatch between {candidate_path} and {parsed_path}")
    candidates = source.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Missing candidates list in {candidate_path}")

    prepared_candidates: list[dict[str, Any]] = []
    changed_count = 0
    seen_candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"Malformed candidate record in {candidate_path}")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in seen_candidate_ids:
            raise ValueError(f"Missing or duplicate candidate ID in {candidate_path}: {candidate_id!r}")
        seen_candidate_ids.add(candidate_id)
        prepared, changed = prepare_candidate(candidate)
        prepared_candidates.append(prepared)
        changed_count += int(changed)

    prepared_episode = copy.deepcopy(source)
    prepared_episode["candidates"] = prepared_candidates
    source_info = parsed.get("source")
    if not isinstance(source_info, dict):
        raise ValueError(f"Missing parsed source metadata in {parsed_path}")
    screenplay_filename = source_info.get("filename")
    if not isinstance(screenplay_filename, str) or Path(screenplay_filename).name != screenplay_filename:
        raise ValueError(f"Invalid original screenplay filename in {parsed_path}")
    screenplay_path = screenplay_directory / screenplay_filename
    if not screenplay_path.is_file():
        raise FileNotFoundError(f"Original screenplay file not found: {screenplay_path}")
    screenplay_bytes = screenplay_path.read_bytes()
    screenplay_digest = _sha256(screenplay_bytes)
    if (
        source_info.get("sha256") != screenplay_digest
        or source_info.get("byte_length") != len(screenplay_bytes)
    ):
        raise ValueError(f"Original screenplay integrity check failed for {screenplay_path}")
    prepared_episode["source_traceability"] = {
        "candidate_file": candidate_path.name,
        "candidate_file_sha256": _sha256(candidate_bytes),
        "parsed_screenplay_file": parsed_path.name,
        "parsed_screenplay_file_sha256": _sha256(parsed_bytes),
        "original_screenplay_file": screenplay_filename,
        "original_screenplay_sha256": screenplay_digest,
        "original_screenplay_byte_length": len(screenplay_bytes),
    }
    prepared_episode["preparation"] = {
        "schema_version": SCHEMA_VERSION,
        "event_order": [
            "start timestamp ascending",
            "end timestamp ascending",
            "source header line ascending, then original input order",
        ],
        "timestamps_preserved": True,
        "event_records_preserved_without_field_changes": True,
        "events_processed": sum(len(candidate["events"]) for candidate in prepared_candidates),
        "candidates_processed": len(prepared_candidates),
        "candidates_reordered": changed_count,
        "candidates_unchanged": len(prepared_candidates) - changed_count,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(prepared_episode, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(prepared_candidates), changed_count, sum(
        len(candidate["events"]) for candidate in prepared_candidates
    )


def prepare_all(
    candidate_directory: Path,
    parsed_directory: Path,
    output_directory: Path,
    screenplay_directory: Path,
) -> dict[str, Any]:
    """Prepare every candidate episode and return aggregate processing counts."""
    if not candidate_directory.is_dir():
        raise NotADirectoryError(f"Candidate directory does not exist: {candidate_directory}")
    if not parsed_directory.is_dir():
        raise NotADirectoryError(f"Parsed screenplay directory does not exist: {parsed_directory}")
    if not screenplay_directory.is_dir():
        raise NotADirectoryError(f"Screenplay directory does not exist: {screenplay_directory}")

    candidate_files = sorted(
        (
            path
            for path in candidate_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".json" and path.name != "summary.json"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not candidate_files:
        raise ValueError(f"No episode candidate JSON files found in {candidate_directory}")

    totals = {"episodes": 0, "candidates": 0, "reordered": 0, "unchanged": 0, "events": 0}
    seen_episode_ids: set[str] = set()
    seen_candidate_ids: set[str] = set()
    for candidate_path in candidate_files:
        parsed_path = parsed_directory / candidate_path.name
        if not parsed_path.is_file():
            raise FileNotFoundError(f"Matching parsed screenplay JSON not found: {parsed_path}")
        candidate_record = _read_json(candidate_path)
        episode_id = candidate_record.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in seen_episode_ids:
            raise ValueError(f"Missing or duplicate episode ID in {candidate_path}: {episode_id!r}")
        seen_episode_ids.add(episode_id)

        candidate_count, changed_count, event_count = _prepare_episode(
            candidate_path,
            parsed_path,
            screenplay_directory,
            output_directory / candidate_path.name,
        )
        episode_candidate_ids = {
            candidate["candidate_id"] for candidate in candidate_record["candidates"]
        }
        duplicates = seen_candidate_ids.intersection(episode_candidate_ids)
        if duplicates:
            raise ValueError(f"Duplicate candidate IDs across episodes: {sorted(duplicates)}")
        seen_candidate_ids.update(episode_candidate_ids)

        totals["episodes"] += 1
        totals["candidates"] += candidate_count
        totals["reordered"] += changed_count
        totals["unchanged"] += candidate_count - changed_count
        totals["events"] += event_count
    return totals


def _project_root() -> Path:
    """Return the repository root from this module's location."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Run the preparer from the command line."""
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Prepare chronologically ordered LLM candidate inputs."
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=root / "data" / "output" / "candidates",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=root / "data" / "output" / "parsed",
    )
    parser.add_argument(
        "--screenplay-dir",
        type=Path,
        default=root / "data" / "input" / "episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "output" / "llm_inputs",
    )
    arguments = parser.parse_args(argv)
    totals = prepare_all(
        arguments.candidate_dir,
        arguments.parsed_dir,
        arguments.output_dir,
        arguments.screenplay_dir,
    )
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
