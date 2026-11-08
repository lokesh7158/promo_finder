"""Safe orchestration and data adapters for the existing CLI pipeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from services.demo_data import synthetic_screenplays
from src.evaluation.llm_client import LLMConfig
from src.evaluation.promo_evaluator import run_proposal_evaluations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_EPISODES = 50
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
STAGES = (
    "Parsing",
    "Candidate generation",
    "LLM input preparation",
    "Promo proposal generation",
    "Evaluation",
    "Ranking",
)
_DEPENDENCIES = {
    "Parsing": (),
    "Candidate generation": ("Parsing",),
    "LLM input preparation": ("Parsing", "Candidate generation"),
    "Promo proposal generation": ("Parsing", "Candidate generation"),
    "Evaluation": ("Parsing", "Candidate generation", "Promo proposal generation"),
    "Ranking": ("Parsing", "Candidate generation", "Promo proposal generation", "Evaluation"),
}


@dataclass
class RunWorkspace:
    """A temporary run directory whose files can be explicitly cleaned."""

    temporary_directory: tempfile.TemporaryDirectory[str]
    root: Path
    stage_seconds: dict[str, float] = field(default_factory=dict)
    stage_logs: list[str] = field(default_factory=list)
    archive_bytes: bytes | None = None

    def cleanup(self) -> None:
        self.temporary_directory.cleanup()


def normalize_stages(selected: Iterable[str]) -> list[str]:
    """Resolve selected pipeline stages and all prerequisite stages in order."""
    selected_set = set(selected)
    unknown = selected_set.difference(STAGES)
    if unknown:
        raise ValueError(f"Unknown pipeline stage(s): {', '.join(sorted(unknown))}")
    required: set[str] = set()

    def include(stage: str) -> None:
        if stage in required:
            return
        required.add(stage)
        for dependency in _DEPENDENCIES[stage]:
            include(dependency)

    for stage in STAGES:
        if stage in selected_set:
            include(stage)
    return [stage for stage in STAGES if stage in required]


def validate_uploads(
    files: Iterable[tuple[str, bytes]],
    *,
    demo_mode: bool,
) -> tuple[list[tuple[str, bytes]], list[str], list[str]]:
    """Validate uploaded names, byte sizes, UTF-8 data, and demo allowlisting."""
    entries = list(files)
    errors: list[str] = []
    warnings: list[str] = []
    if not entries:
        return [], ["Upload at least one .txt screenplay file."], warnings
    if len(entries) > MAX_EPISODES:
        errors.append(f"Upload at most {MAX_EPISODES} files; received {len(entries)}.")

    expected_demo = (
        {
            hashlib.sha256(content).hexdigest(): (name, content)
            for name, content in synthetic_screenplays()
        }
        if demo_mode
        else {}
    )
    valid: list[tuple[str, bytes]] = []
    seen_names: set[str] = set()
    total_bytes = 0
    for original_name, content in entries:
        name = Path(original_name.replace("\\", "/")).name
        if name != original_name.replace("\\", "/") or name in {"", ".", ".."}:
            errors.append("A filename contains an unsupported path component.")
            continue
        name_key = name.casefold()
        if name_key in seen_names:
            warnings.append(f"Duplicate filename: {name}")
            errors.append(f"Duplicate filenames are not allowed: {name}")
            continue
        seen_names.add(name_key)
        if Path(name).suffix.casefold() != ".txt":
            errors.append(f"{name}: file type must be .txt.")
            continue
        if not content:
            errors.append(f"{name}: file is empty.")
            continue
        if len(content) > MAX_FILE_BYTES:
            errors.append(f"{name}: exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB per-file limit.")
            continue
        total_bytes += len(content)
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name}: screenplay must be UTF-8 encoded.")
            continue
        if demo_mode:
            digest = hashlib.sha256(content).hexdigest()
            allowed = expected_demo.get(digest)
            if allowed is None or name != allowed[0]:
                errors.append(
                    f"{name}: hosted demo accepts only unchanged files from the synthetic sample pack."
                )
                continue
        valid.append((name, content))

    if total_bytes > MAX_TOTAL_BYTES:
        errors.append(f"Combined upload exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB.")
    if len(entries) > MAX_EPISODES:
        valid = []
    return valid, errors, warnings


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def _iter_json_array(path: Path, key: str) -> Iterable[dict[str, Any]]:
    """Decode one top-level JSON array incrementally to bound memory use."""
    decoder = json.JSONDecoder()
    marker = f'"{key}"'
    with path.open("r", encoding="utf-8") as file:
        buffer = ""
        position = 0
        while marker not in buffer:
            buffer += file.read(64 * 1024)
            if not buffer:
                raise ValueError(f"Missing JSON array {key!r} in {path.name}")
        position = buffer.index(marker) + len(marker)
        while True:
            if position >= len(buffer):
                chunk = file.read(64 * 1024)
                if not chunk:
                    raise ValueError(f"Malformed JSON array {key!r} in {path.name}")
                buffer = buffer[position:] + chunk
                position = 0
            if buffer[position] in " \t\r\n:":
                position += 1
                continue
            if buffer[position] == "[":
                position += 1
                break
            raise ValueError(f"Malformed JSON array header {key!r} in {path.name}")

        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                chunk = file.read(64 * 1024)
                if not chunk:
                    raise ValueError(f"Malformed JSON array item in {path.name}")
                buffer = buffer[position:] + chunk
                position = 0
                continue
            if not isinstance(item, dict):
                raise ValueError(f"Expected object items in {key!r} from {path.name}")
            yield item
            position = end
            if position > 1024 * 1024:
                buffer = buffer[position:]
                position = 0


def _run_cli(
    workspace: RunWorkspace,
    stage: str,
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    if progress:
        progress(f"Running {stage.lower()}...")
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30 * 60,
        check=False,
    )
    elapsed = time.perf_counter() - started
    workspace.stage_seconds[stage] = elapsed
    if result.returncode:
        workspace.stage_logs.append(f"{stage} failed (exit code {result.returncode}).")
        raise RuntimeError(f"{stage} failed (exit code {result.returncode}); output was withheld.")
    workspace.stage_logs.append(f"{stage} completed in {elapsed:.2f} seconds.")
    if progress:
        progress(f"{stage} completed in {elapsed:.2f}s.")


def run_pipeline(
    files: list[tuple[str, bytes]],
    *,
    selected_stages: Iterable[str],
    run_gemini: bool,
    llm_environment: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunWorkspace:
    """Run selected existing CLI stages in a fresh temporary workspace."""
    stages = normalize_stages(selected_stages)
    if not stages:
        raise ValueError("Select at least one pipeline stage.")
    if not files:
        raise ValueError("No uploaded screenplay files were provided.")
    temp = tempfile.TemporaryDirectory(prefix="promofinder-run-")
    workspace = RunWorkspace(temp, Path(temp.name))
    try:
        source_dir = workspace.root / "inputs"
        source_dir.mkdir()
        for name, content in files:
            (source_dir / name).write_bytes(content)
        output = workspace.root / "outputs"
        parsed = output / "parsed"
        candidates = output / "candidates"
        llm_inputs = output / "llm_inputs"
        proposals = output / "promo_proposals"
        evaluations = output / "proposal_evaluations"

        if "Parsing" in stages:
            _run_cli(
                workspace,
                "Parsing",
                [
                    str(PROJECT_ROOT / "src" / "ingestion" / "screenplay_parser.py"),
                    "--input-dir", str(source_dir), "--output-dir", str(parsed),
                ],
                progress=progress,
            )
        if "LLM input preparation" not in stages:
            shutil.rmtree(source_dir)
        if "Candidate generation" in stages:
            _run_cli(
                workspace,
                "Candidate generation",
                [
                    str(PROJECT_ROOT / "src" / "candidate_generation" / "candidate_generator.py"),
                    "--input-dir", str(parsed), "--output-dir", str(candidates),
                ],
                progress=progress,
            )
        if "LLM input preparation" in stages:
            _run_cli(
                workspace,
                "LLM input preparation",
                [
                    str(PROJECT_ROOT / "src" / "evaluation" / "llm_input_preparer.py"),
                    "--candidate-dir", str(candidates),
                    "--parsed-dir", str(parsed),
                    "--screenplay-dir", str(source_dir),
                    "--output-dir", str(llm_inputs),
                ],
                progress=progress,
            )
            shutil.rmtree(source_dir)
        if "Promo proposal generation" in stages:
            _run_cli(
                workspace,
                "Promo proposal generation",
                [
                    str(PROJECT_ROOT / "src" / "promo_construction" / "promo_proposal_generator.py"),
                    "--parsed-dir", str(parsed),
                    "--candidate-dir", str(candidates),
                    "--output-dir", str(proposals),
                ],
                progress=progress,
            )
        if "Evaluation" in stages:
            command = [
                str(PROJECT_ROOT / "src" / "evaluation" / "promo_evaluator.py"),
                "--proposal-input", str(proposals / "shortlist.json"),
                "--parsed-dir", str(parsed),
                "--candidate-dir", str(candidates),
                "--output-dir", str(evaluations),
            ]
            if not run_gemini:
                command.append("--dry-run")
            environment = os.environ.copy()
            environment.update(llm_environment or {})
            if run_gemini:
                configured_provider = environment.get("PROMO_LLM_PROVIDER", "gemini")
                if configured_provider.strip().lower() != "gemini":
                    raise ValueError("This application only enables the configured Gemini provider.")
                environment["PROMO_LLM_PROVIDER"] = "gemini"
                environment.setdefault("PROMO_LLM_MODEL", "gemini-3.5-flash-lite")
                if environment.get("PROMO_GEMINI_FREE_TIER_CONFIRMED") != "YES":
                    raise ValueError("Gemini evaluation requires explicit free-tier confirmation.")
                if not environment.get("GEMINI_API_KEY"):
                    raise ValueError("Set GEMINI_API_KEY in the local environment or Streamlit secrets.")
            _run_cli(
                workspace, "Evaluation", command, environment=environment, progress=progress
            )
        if "Ranking" in stages and not (evaluations / "ranked_proposals.json").is_file():
            workspace.stage_logs.append(
                "Ranking unavailable: offline dry-run validates inputs but does not create evaluation scores."
            )
        return workspace
    except Exception:
        workspace.cleanup()
        raise


def run_live_evaluation(
    workspace: RunWorkspace,
    *,
    config: LLMConfig,
    limit: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run a bounded live proposal evaluation in the current temporary workspace."""
    if limit < 1:
        raise ValueError("Maximum proposals to evaluate must be at least one.")
    output = workspace.root / "outputs"
    proposal_file = output / "promo_proposals" / "shortlist.json"
    parsed = output / "parsed"
    candidates = output / "candidates"
    evaluations = output / "proposal_evaluations"
    if not proposal_file.is_file():
        raise ValueError("Run proposal generation before starting LLM evaluation.")
    started = time.perf_counter()
    result = run_proposal_evaluations(
        proposal_file=proposal_file,
        parsed_directory=parsed,
        candidate_directory=candidates,
        output_directory=evaluations,
        config=config,
        limit=limit,
        progress=progress,
    )
    elapsed = time.perf_counter() - started
    workspace.stage_seconds["Evaluation"] = (
        workspace.stage_seconds.get("Evaluation", 0.0) + elapsed
    )
    workspace.stage_logs.append(
        f"Live evaluation finished: {result['validated']} validated, "
        f"{result['failed']} failed, provider {config.provider}, model {config.model}."
    )
    return result


def provider_usage_summary(
    workspace: RunWorkspace,
    *,
    provider: str | None = None,
) -> dict[str, Any]:
    """Aggregate only token fields returned by provider APIs; do not estimate cost."""
    raw_directory = workspace.root / "outputs" / "proposal_evaluations" / "raw_responses"
    totals: Counter[str] = Counter()
    requests = 0
    if raw_directory.is_dir():
        for path in raw_directory.glob("*.json"):
            record = _read_json(path)
            metadata = record.get("provider_metadata", {})
            if provider is not None and (
                not isinstance(metadata, dict) or metadata.get("provider") != provider
            ):
                continue
            usage = metadata.get("usage") if isinstance(metadata, dict) else None
            if not isinstance(usage, dict):
                continue
            requests += 1
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "promptTokenCount",
                "candidatesTokenCount",
                "totalTokenCount",
            ):
                value = usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[name] += value
    return {"requests_with_usage": requests, "reported_token_counts": dict(totals)}


def run_metrics(workspace: RunWorkspace) -> dict[str, Any]:
    """Aggregate run metrics from the existing stage output schemas."""
    output = workspace.root / "outputs"
    parsed_files = sorted((output / "parsed").glob("*.json")) if (output / "parsed").exists() else []
    episodes = []
    scenes_count = 0
    events_count = 0

    def timestamped_event_count(scenes: list[Any]) -> int:
        return sum(
            1
            for scene in scenes
            if isinstance(scene, dict)
            for event in scene.get("events", [])
            if isinstance(event, dict)
            and event.get("start", {}).get("valid") is True
            and event.get("end", {}).get("valid") is True
        )

    for path in parsed_files:
        episode = _read_json(path)
        scenes = episode.get("scenes", [])
        scenes_count += len(scenes)
        episode_event_count = timestamped_event_count(scenes)
        events_count += episode_event_count
        number_value = episode.get("metadata", {}).get("episode_number", {}).get("value", "")
        episode_number_match = re.search(r"\d+", str(number_value))
        episodes.append(
            {
                "episode": str(number_value).strip() or path.stem,
                "episode_id": episode.get("episode_id", path.stem),
                "episode_number": (
                    episode_number_match.group(0) if episode_number_match else path.stem
                ),
                "scene_count": len(scenes),
                "event_count": episode_event_count,
                "parse_status": episode.get("validation", {}).get("status", "unknown"),
            }
        )

    candidate_summary_path = output / "candidates" / "summary.json"
    candidate_summary = _read_json(candidate_summary_path) if candidate_summary_path.is_file() else {}
    candidate_count = candidate_summary.get("counts", {}).get("stage_b_candidates", 0)
    candidate_by_episode = {
        str(row.get("episode_id", "")): len(row.get("candidates", []))
        for path in (output / "candidates").glob("*.json")
        if path.name != "summary.json"
        for row in [_read_json(path)]
    } if (output / "candidates").exists() else {}
    for episode in episodes:
        episode["candidate_count"] = candidate_by_episode.get(episode["episode_id"], 0)

    proposal_summary_path = output / "promo_proposals" / "summary.json"
    proposal_summary = _read_json(proposal_summary_path) if proposal_summary_path.is_file() else {}
    summary = proposal_summary.get("summary", {})
    proposal_count = summary.get("proposal_count_before_validation", 0)
    proposal_by_episode: Counter[str] = Counter()
    all_proposals_path = output / "promo_proposals" / "all_proposals.json"
    if all_proposals_path.is_file():
        for proposal in _iter_json_array(all_proposals_path, "proposals"):
            episode_numbers = {
                str(segment.get("episode_id"))
                for segment in proposal.get("segments", [])
                if isinstance(segment, dict) and segment.get("episode_id") is not None
            }
            proposal_by_episode.update(episode_numbers)
    for episode in episodes:
        episode["proposal_count"] = proposal_by_episode.get(episode["episode_id"], 0)

    shortlist_path = output / "promo_proposals" / "shortlist.json"
    shortlist = _read_json(shortlist_path).get("proposals", []) if shortlist_path.is_file() else []
    ranked_path = output / "proposal_evaluations" / "ranked_proposals.json"
    ranked_document = _read_json(ranked_path) if ranked_path.is_file() else {}
    ranked_records = ranked_document.get("overall_ranking", [])
    evaluation_summary_path = output / "proposal_evaluations" / "proposal_evaluation_summary.json"
    if not evaluation_summary_path.is_file():
        evaluation_summary_path = (
            output / "proposal_evaluations" / "proposal_dry_run_summary.json"
        )
    evaluation_summary = (
        _read_json(evaluation_summary_path) if evaluation_summary_path.is_file() else {}
    )
    evaluation_count = evaluation_summary.get("validated_proposal_count", len(ranked_records))
    evaluations = {
        record.get("promo_id"): record
        for record in ranked_records
        if isinstance(record, dict) and record.get("promo_id")
    }
    results = []
    for proposal in shortlist:
        evaluation = evaluations.get(proposal.get("promo_id"), {})
        results.append({**proposal, "evaluation": evaluation})
    for rank, record in enumerate(ranked_records, start=1):
        record["rank"] = rank
        proposal_id = record.get("promo_id")
        matching = next((p for p in shortlist if p.get("promo_id") == proposal_id), None)
        if matching is not None:
            merged = {**matching, "evaluation": record, "rank": rank}
            existing = next((i for i, item in enumerate(results) if item.get("promo_id") == proposal_id), None)
            if existing is None:
                results.append(merged)
            else:
                results[existing] = merged
    return {
        "uploaded_episodes": len(parsed_files),
        "parsed_episodes": sum(
            episode["parse_status"] in {"valid", "warnings"} for episode in episodes
        ),
        "scenes": scenes_count,
        "events": events_count,
        "candidates": candidate_count,
        "proposals": proposal_count,
        "evaluations": evaluation_count,
        "classification_distribution": ranked_document.get("classification_distribution", {}),
        "status": "complete" if parsed_files else "not_run",
        "episodes": episodes,
        "shortlist": shortlist,
        "results": results,
        "evaluation_summary": evaluation_summary,
        "candidate_summary": candidate_summary,
        "proposal_summary": proposal_summary,
        "stage_seconds": dict(workspace.stage_seconds),
        "logs": list(workspace.stage_logs),
    }


def analytics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build an exportable aggregate analytics view from actual run outputs."""
    results = metrics.get("results", [])
    scores = [
        result["evaluation"].get("final_score")
        for result in results
        if isinstance(result.get("evaluation"), dict)
        and isinstance(result["evaluation"].get("final_score"), (int, float))
    ]
    category_counts = Counter(result.get("promo_category", "UNKNOWN") for result in results)
    duration_bands = Counter()
    for result in results:
        duration = result.get("total_duration_seconds")
        if isinstance(duration, (int, float)):
            label = "<30s" if duration < 30 else "30-45s" if duration < 45 else "45-60s"
            duration_bands[label] += 1
    return {
        "schema_version": 1,
        "source": "current_execution",
        "uploaded_episodes": metrics.get("uploaded_episodes", 0),
        "parsed_episodes": metrics.get("parsed_episodes", 0),
        "scene_count": metrics.get("scenes", 0),
        "event_count": metrics.get("events", 0),
        "candidate_count": metrics.get("candidates", 0),
        "proposal_count": metrics.get("proposals", 0),
        "evaluation_count": metrics.get("evaluations", 0),
        "classification_distribution": metrics.get("classification_distribution", {}),
        "category_counts_in_results": dict(category_counts),
        "duration_bands_in_results": dict(duration_bands),
        "evaluation_score_count": len(scores),
        "evaluation_score_average": sum(scores) / len(scores) if scores else None,
        "episode_statistics": metrics.get("episodes", []),
        "processing_seconds_by_stage": metrics.get("stage_seconds", {}),
    }


def ranked_records(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Return displayed result rows in existing ranking order or shortlist order."""
    rows = list(metrics.get("results", []))
    rows.sort(
        key=lambda item: (
            item.get("rank") is None,
            item.get("rank", item.get("shortlist_rank", 10**9)),
        )
    )
    return rows


def records_csv(rows: list[dict[str, Any]]) -> bytes:
    """Serialize visible promo records as a portable CSV table."""
    output = io.StringIO(newline="")
    columns = [
        "rank", "promo_id", "episode_numbers", "promo_category", "total_duration_seconds",
        "final_score", "classification", "human_review_required",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        evaluation = row.get("evaluation", {})
        episode_numbers = sorted(
            {
                segment.get("episode_number")
                for segment in row.get("segments", [])
                if segment.get("episode_number") is not None
            }
        )
        writer.writerow(
            {
                "rank": row.get("rank", row.get("shortlist_rank")),
                "promo_id": row.get("promo_id"),
                "episode_numbers": ",".join(map(str, episode_numbers)),
                "promo_category": row.get("promo_category"),
                "total_duration_seconds": row.get("total_duration_seconds"),
                "final_score": evaluation.get("final_score"),
                "classification": evaluation.get("classification"),
                "human_review_required": evaluation.get("human_review_required"),
            }
        )
    return output.getvalue().encode("utf-8-sig")


def outputs_zip(workspace: RunWorkspace) -> bytes:
    """Package generated pipeline outputs without the uploaded source directory."""
    output = io.BytesIO()
    output_root = workspace.root / "outputs"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root).as_posix())
    return output.getvalue()


def historical_results(root: Path = PROJECT_ROOT) -> dict[str, Any] | None:
    """Read local historical outputs only; caller must never use this in demo mode."""
    output = root / "data" / "output"
    ranking_path = output / "promo_selection" / "ranked_189.json"
    if not ranking_path.is_file():
        return None
    ranking = _read_json(ranking_path)
    candidate_path = output / "candidates" / "summary.json"
    proposal_path = output / "promo_proposals" / "summary.json"
    consolidated_path = output / "proposal_evaluations" / "consolidated-189" / "manifest.json"
    historical_parsed_dir = output / "parsed"
    historical_scene_count = 0
    historical_event_count = 0
    if historical_parsed_dir.is_dir():
        for parsed_path in historical_parsed_dir.glob("*.json"):
            episode = _read_json(parsed_path)
            scenes = episode.get("scenes", [])
            historical_scene_count += len(scenes)
            historical_event_count += sum(
                sum(
                    1
                    for event in scene.get("events", [])
                    if isinstance(event, dict)
                    and event.get("start", {}).get("valid") is True
                    and event.get("end", {}).get("valid") is True
                )
                for scene in scenes
                if isinstance(scene, dict) and isinstance(scene.get("events"), list)
            )
    final_recommendations = []
    readme_path = root / "README.md"
    if readme_path.is_file():
        in_final_five_section = False
        columns = (
            "Post",
            "Episode / proposal ID",
            "Source range(s)",
            "Duration",
            "Screenplay hook to ending",
            "Editorial status",
        )
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## 10. Human review and final five recommendations"):
                in_final_five_section = True
                continue
            if in_final_five_section and line.startswith("## "):
                break
            if not in_final_five_section or not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != len(columns) or not cells[0].isdigit():
                continue
            final_recommendations.append(dict(zip(columns, cells, strict=True)))
    return {
        "ranking": ranking,
        "candidate_summary": _read_json(candidate_path) if candidate_path.is_file() else {},
        "proposal_summary": _read_json(proposal_path) if proposal_path.is_file() else {},
        "consolidated_manifest": (
            _read_json(consolidated_path) if consolidated_path.is_file() else {}
        ),
        "scene_count": historical_scene_count,
        "event_count": historical_event_count,
        "final_recommendations": final_recommendations,
        "source": "data/output (read-only historical data)",
    }


def load_historical_results(*, demo_mode: bool, root: Path = PROJECT_ROOT) -> dict[str, Any] | None:
    """Enforce the rule that demo mode cannot access local historical output."""
    if demo_mode:
        return None
    return historical_results(root)
